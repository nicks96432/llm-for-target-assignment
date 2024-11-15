"""
Generate dataset for training.

TODO: add documentation
"""

import hashlib
import logging
import os
import sys
import tempfile
from argparse import ArgumentParser
from typing import IO

import geopandas
import requests
from geopandas import GeoDataFrame
from matplotlib import pyplot
from numpy import random
from pandas import DataFrame, RangeIndex
from shapely import Point, Polygon, shortest_line
from shapely.plotting import plot_points, plot_polygon
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


def download_taiwan_map_data(cache: bool = True) -> IO[bytes]:
    """
    Download Taiwan map from MOI.

    Returns
    -------
    IO[bytes]
        the downloaded zip file

    Raises
    ------
    ValueError
        if sha256 checksum does not match
    """

    TAIWAN_MAP_ZIP_URL = (
        "https://data.moi.gov.tw/MoiOD/System/DownloadFile.aspx"
        "?DATA=72874C55-884D-4CEA-B7D6-F60B0BE85AB0"
    )
    TAIWAN_MAP_ZIP_FILENAME = "直轄市、縣(市)界線檔(TWD97經緯度)1130719.zip"
    TAIWAN_MAP_ZIP_SHA256 = (
        "292b1fdcc5dcd9e4e9ddbeb49599224c0222268fd6f9d7360247310aa3082ad0"
    )

    if cache:
        try:
            f = open(TAIWAN_MAP_ZIP_FILENAME, "rb")
        except FileNotFoundError:
            pass
        else:
            logger.info("found Taiwan map in cache")

            file_hash = hashlib.file_digest(f, "sha256").hexdigest()

            if file_hash == TAIWAN_MAP_ZIP_SHA256:
                logger.info("cache file sha256 checksum matches")
                return f

        logger.info("cache file sha256 checksum does not match, re-downloading")
        f = open(TAIWAN_MAP_ZIP_FILENAME, "w+b")
    else:
        logger.info("not using cache, downloading Taiwan map")
        f = tempfile.NamedTemporaryFile(suffix=".zip")

    with requests.get(TAIWAN_MAP_ZIP_URL, stream=True) as response:
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", 0))

        bar_args = {
            "desc": "downloading Taiwan map",
            "miniters": 1,
            "total": size,
            "unit_divisor": 1024,
            "unit_scale": True,
            "unit": "B",
        }

        with tqdm(**bar_args) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    f.seek(0)
    zip_hash = hashlib.file_digest(f, "sha256").hexdigest()  # type: ignore

    if zip_hash != TAIWAN_MAP_ZIP_SHA256:
        msg = "SHA256 checksum does not match"
        raise ValueError(msg)

    return f


def get_taiwan_polygon(
    cache: bool = True,
    simplify_tol: float = 1e-3,
) -> GeoDataFrame:
    """
    Get Taiwan island polygon based on MOI data.

    Parameters
    ----------
    cache : bool, optional
        whether to cache the downloaded data, by default True
    simplify_tol : float, optional
        polygon simplification tolerance, by default 1e-3

    Returns
    -------
    GeoDataFrame
        Taiwan island polygon
    """
    TAIWAN_MAP_FILENAME = "COUNTY_MOI_1130718.shp"
    with download_taiwan_map_data(cache) as data_zip:
        taiwan_map: GeoDataFrame = geopandas.read_file(
            f"zip://{data_zip.name}/{TAIWAN_MAP_FILENAME}"
        )

    # dissolve into a MultiPolygon
    taiwan_map["tmp"] = 0
    dissolved_taiwan_map = taiwan_map.dissolve(by="tmp")

    # select the largest polygon, i.e. Taiwan island
    taiwan = GeoDataFrame(
        {
            "geometry": [
                max(
                    dissolved_taiwan_map["geometry"].values[0].geoms,
                    key=lambda x: x.area,
                )
            ]
        },
        crs=dissolved_taiwan_map.crs,
    )  # type: ignore

    return taiwan.simplify(simplify_tol, preserve_topology=False)


def generate_missles(rng: random.Generator, n: int) -> DataFrame:
    """
    Generate missle types

    Parameters
    ----------
    rng : random.Generator
        random number generator
    n : int
        missle type count

    Returns
    -------
    DataFrame
        missle types
    """

    min_ranges = rng.integers(5000, 10000, n)
    min_ranges.sort()
    max_ranges = rng.integers(50000, 100000, n)
    max_ranges.sort()

    missles = DataFrame(
        {
            "min_range": min_ranges,
            "max_range": max_ranges,
        }
    )
    missles.index = RangeIndex(1, n + 1)
    missles.index.name = "type"

    return missles


def generate_ship_types(rng: random.Generator, m: int, n: int) -> DataFrame:
    """
    Generate ship types

    Parameters
    ----------
    rng : random.Generator
        random number generator
    m : int
        ship type count
    n : int
        missle type count

    Returns
    -------
    DataFrame
        ship types
    """

    ship_types = DataFrame(
        {
            f"require_missle_{i}_count": rng.choice([*list(range(1, m + 1)), -1], m)
            for i in range(1, n + 1)
        }
    )

    ship_types.index = RangeIndex(1, m + 1)
    ship_types.index.name = "type"

    return ship_types


def generate_ships(
    dataframe: GeoDataFrame,
    rng: random.Generator,
    ship_count: int,
    max_ship_distance: int,
    m: int,
    twd97: bool = False,
) -> GeoDataFrame:
    """
    Generate ships in the specified dataframe.

    Parameters
    ----------
    dataframe : GeoDataFrame
        dataframe representing the island where ships are attacking
    rng : random.Generator
        random number generator
    ship_count : int
        number of ships
    max_ship_distance : int
        maximum distance between ships
    m : int
        ship type count
    twd97 : bool, optional
        if True, generate ships in TWD97 coordinate system, by default False

    Returns
    -------
    GeoDataFrame
        ships
    """

    # convert to TWD97 (epsg: 3826)
    dataframe = dataframe.to_crs(epsg=3826)

    polygon: Polygon = dataframe.iloc[0]
    radar = polygon.buffer(max_ship_distance)
    min_x, min_y, max_x, max_y = radar.bounds

    ships = []
    count = 0
    while count < ship_count:
        random_point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if random_point.within(radar) and not random_point.within(polygon):
            ships.append(random_point)
            count += 1

    ships = GeoDataFrame(
        {"geometry": ships, "type": rng.integers(1, m + 1, len(ships))},
        crs=dataframe.crs,
    )  # type: ignore
    ships.index = RangeIndex(1, ship_count + 1)
    ships.index.name = "id"

    if twd97:
        return ships

    return ships.to_crs(epsg=4326)  # type: ignore


def generate_turrets(
    dataframe: GeoDataFrame,
    ship_types: DataFrame,
    ships: GeoDataFrame,
    missles: DataFrame,
    rng: random.Generator,
    turret_count: int,
    max_turret_coast_distance: int,
    twd97: bool = False,
) -> GeoDataFrame:
    """
    Generate turrets in the specified dataframe.

    Parameters
    ----------
    dataframe : GeoDataFrame
        dataframe representing the island where turrets are
        generated
    ship_types : DataFrame
        ship types
    ships : GeoDataFrame
        ships
    missles : DataFrame
        missle types
    rng : random.Generator
        random number generator
    turret_count : int
        number of turrets
    max_turret_coast_distance : int
        maximum distance between turrets and coast
    n : int
        missle type count
    twd97 : bool, optional
        if True, generate turrets in TWD97 coordinate system, by default False

    Returns
    -------
    GeoDataFrame
        turrets
    """

    # convert to TWD97 (epsg: 3826)
    dataframe = dataframe.to_crs(epsg=3826)  # type: ignore
    ships = ships.to_crs(epsg=3826)  # type: ignore

    polygon = dataframe.iloc[0]

    turrets = []

    count = 0
    for _, ship in ships.iterrows():
        # select the missle types that can damage the ship
        missle_damages = ship_types.loc[ship.type]
        missle_damages.rename(
            {f"require_missle_{i}_count": i for i in missles.index}, inplace=True
        )
        missle_damages = missle_damages[missle_damages != -1]

        missle = missles.loc[rng.choice(missles.index)]

        max_range: Polygon = ship.geometry.buffer(missle.max_range).intersection(
            polygon
        )
        min_range: Polygon = ship.geometry.buffer(missle.min_range).intersection(
            polygon
        )
        allowed_range = max_range.difference(min_range)
        min_x, min_y, max_x, max_y = allowed_range.bounds

        while True:
            random_point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if (
                random_point.within(allowed_range)
                and shortest_line(random_point, polygon.boundary).length
                < max_turret_coast_distance
            ):
                turrets.append(
                    {
                        "geometry": random_point,
                        f"missle_{missle.name}_count": missle_damages[missle.name],
                    }
                )

                count += 1
                break

    while count < turret_count:
        random_point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if (
            random_point.within(polygon)
            and shortest_line(random_point, polygon.boundary).length
            < max_turret_coast_distance
        ):
            turrets.append({"geometry": random_point})
            count += 1

    for turret in turrets:
        for i in missles.index:
            if not turret.get(f"missle_{i}_count"):
                turret[f"missle_{i}_count"] = rng.integers(0, 10)

    turrets = GeoDataFrame(turrets, crs=dataframe.crs)  # type: ignore
    turrets.index = RangeIndex(1, turret_count + 1)
    turrets.index.name = "id"

    if twd97:
        return turrets

    return turrets.to_crs(epsg=4326)  # type: ignore


def point_to_xy(df: DataFrame) -> DataFrame:
    df = df.assign(x=df.geometry.x, y=df.geometry.y)
    df.drop("geometry", axis=1, inplace=True)

    return df


def generate_dataset(
    n: int = 2,
    seed: int = 0,
    turret_count: int = 5,
    max_turret_coast_distance: int = 3000,
    m: int = 10,
    ship_count: int = 5,
    max_ship_distance: int = 3000,
    log_level: int = logging.WARNING,
    preview: bool = False,
    twd97: bool = False,
    output_dir: str = ".",
) -> None:
    rng = random.Generator(random.PCG64DXSM(seed))

    logger.setLevel(log_level)

    taiwan = get_taiwan_polygon()

    missles = generate_missles(rng=rng, n=n)
    ship_types = generate_ship_types(rng=rng, m=m, n=n)
    ships = generate_ships(
        dataframe=taiwan,
        rng=rng,
        ship_count=ship_count,
        max_ship_distance=max_ship_distance,
        m=m,
        twd97=twd97,
    )
    turrets = generate_turrets(
        dataframe=taiwan,
        ship_types=ship_types,
        ships=ships,
        missles=missles,
        rng=rng,
        turret_count=turret_count,
        max_turret_coast_distance=max_turret_coast_distance,
        twd97=twd97,
    )

    os.chdir(output_dir)

    if preview:
        taiwan: GeoDataFrame = taiwan.to_crs(epsg=3826)  # type: ignore
        plot_polygon(taiwan.geometry[0], add_points=False)
        plot_points(turrets.geometry, color="blue", marker=".")
        plot_points(ships.geometry, color="red", marker=".")
        pyplot.savefig("preview.png")

    point_to_xy(turrets).to_csv("turrets.csv")
    point_to_xy(ships).to_csv("ships.csv")

    missles.to_csv("missles.csv")
    ship_types.to_csv("ship_types.csv")


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
        default=logging.WARNING,
    )
    arg_parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    arg_parser.add_argument(
        "-O",
        "--output-dir",
        help="dataset output directory",
        type=str,
        default=".",
    )
    arg_parser.add_argument(
        "--seed",
        help="random seed",
        type=int,
        default=0,
    )
    arg_parser.add_argument(
        "--preview",
        help="preview generated dataset on map",
        action="store_true",
        default=False,
    )
    arg_parser.add_argument(
        "--twd97",
        help="use TWD97 coordinate system",
        action="store_true",
        default=False,
    )

    arg_parser.add_argument(
        "--max-turret-coast-distance",
        help="max turret-coast distance in meters",
        type=int,
        default=3000,
    )
    arg_parser.add_argument(
        "--turret-count",
        help="number of turrets",
        type=int,
        default=5,
    )

    arg_parser.add_argument(
        "-n",
        "--n",
        help="number of missle types",
        type=int,
        default=3,
    )

    arg_parser.add_argument(
        "-m",
        "--m",
        help="number of battleship types",
        type=int,
        default=10,
    )
    arg_parser.add_argument(
        "--ship-count",
        help="number of battleships",
        type=int,
        default=5,
    )
    arg_parser.add_argument(
        "--max-ship-distance",
        help="maximum battleship distance in meters",
        type=int,
        default=30000,
    )

    generate_dataset(**arg_parser.parse_args().__dict__)
