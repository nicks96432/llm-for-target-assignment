"""
Generate dataset for training.
"""

import dataclasses
import hashlib
import inspect
import json
import logging
import os
import pathlib
import sys
import tempfile
from argparse import ArgumentParser
from typing import IO, Self

import geopandas
import numpy
import pandas
import requests
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot
from numpy import random
from pandas import RangeIndex
from shapely import Point, Polygon, shortest_line
from shapely.plotting import plot_points, plot_polygon
from tqdm.auto import tqdm

from config import DatasetConfig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


def geometry_to_xy(df: GeoDataFrame) -> GeoDataFrame:
    """
    Converts geometry in a `GeoDataFrame` to x and y columns

    Parameters
    ----------
    df : GeoDataFrame
        a dataframe with a `geometry` attribute to convert

    Returns
    -------
    GeoDataFrame
        a dataframe with x and y columns
    """

    df = df.assign(x=df.geometry.x, y=df.geometry.y)

    return df


@dataclasses.dataclass(frozen=True)
class Missle:
    """
    Dataclass for a missle.
    """

    id: int
    type: int
    turret_id: int
    available_targets: list[int]


class Dataset:
    """
    A dataset of ships, turrets, and missles.

    Attributes
    ----------
    distances : `numpy.ndarray`
        array of shape `(n_turret, n_ship)` representing distances between each
        turret and each ship
    in_range : `numpy.ndarray`
        array of shape `(n_missle_type, n_turret, n_ship)` indicating whether a ship
        is in range of a turret for a given missile type
    missle_damages : `numpy.ndarray`
        array of shape `(n_ship_type, n_missle_type)` indicating the damage a missile
        type can inflict on each ship type
    missle_types : `pandas.DataFrame`
        dataframe with columns `type`, `min_range` and `max_range`
    missles : `list[Missle]`
        list of missles
    require_missles : `numpy.ndarray`
        array of shape `(n_ship_type, n_missle_type)` indicating how many missiles
        are required to destroy each ship type with each missile type
    ships_xy : `numpy.ndarray`
        array of shape `(n_ships, 2)` containing the x and y coordinates of each ship
    ship_types: `numpy.ndarray`
        array of shape `(n_ship)` indicating the type of each ship
    turrets_missle_count : `numpy.ndarray`
        array of shape `(n_turret, n_missle_type)` representing the x and y coordinates
        of each turret
    turrets_xy : `numpy.ndarray`
        array of shape `(n_turret, 2)` representing the x and y coordinates of each
        turret

    Note
    ----
    IDs for ships, turrets, and missile types are indexed from 1 based on their order in
    the corresponding data structure.
    """

    __slots__ = [
        "distances",
        "in_range",
        "missle_damages",
        "missle_types",
        "missles",
        "require_missles",
        "ship_types",
        "ships_xy",
        "turrets_missle_count",
        "turrets_xy",
    ]

    @staticmethod
    def _download_taiwan_map_data(cache: bool = True) -> IO[bytes]:
        """
        Downloads Taiwan map from MOI.

        Parameters
        ----------
        cache : bool, optional
            whether to cache the downloaded zip file, by default True

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
            "https://data.moi.gov.tw/MoiOD/System/DownloadFile.aspx?DATA=72874C55-884D-4CEA-B7D6-F60B0BE85AB0"
        )
        TAIWAN_MAP_ZIP_FILENAME = "直轄市、縣(市)界線1140318.zip"
        TAIWAN_MAP_ZIP_SHA256 = (
            "9ff4d337e03a711d7e17caaf4d0c46a0fe2c6bd3660b321a2e35123030d8491c"
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

    @staticmethod
    def _get_taiwan_polygon(
        cache: bool = True,
        simplify_tol: float = 1e-3,
    ) -> GeoSeries:
        """
        Gets Taiwan island polygon based on MOI data.

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
        TAIWAN_MAP_FILENAME = "COUNTY_MOI_1140318.shp"
        with Dataset._download_taiwan_map_data(cache) as data_zip:
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

    @staticmethod
    def _generate_missle_types(
        rng: random.Generator, n_missle_type: int
    ) -> pandas.DataFrame:
        """
        Generates missle types.

        Parameters
        ----------
        rng : random.Generator
            random number generator
        n_missle_type : int
            missle type count

        Returns
        -------
        pandas.DataFrame
            missle types
        """

        min_ranges = rng.integers(5000, 10000, n_missle_type)
        min_ranges.sort()
        max_ranges = rng.integers(50000, 100000, n_missle_type)
        max_ranges.sort()

        missles = pandas.DataFrame({
            "min_range": min_ranges,
            "max_range": max_ranges,
        })
        missles["type"] = RangeIndex(1, n_missle_type + 1)

        return missles

    @staticmethod
    def _generate_ship_types(
        rng: random.Generator, n_ship_type: int, n_missle_type: int
    ) -> pandas.DataFrame:
        """
        Generates ship types.

        Parameters
        ----------
        rng : random.Generator
            random number generator
        n_ship_type : int
            ship type count
        n_missle_type : int
            missle type count

        Returns
        -------
        pandas.DataFrame
            ship types
        """

        ship_types = pandas.DataFrame({
            f"require_missle_{i}_count": rng.choice(
                [*list(range(1, n_ship_type + 1)), -1], n_ship_type
            )
            for i in range(1, n_missle_type + 1)
        })

        ship_types["type"] = RangeIndex(1, n_ship_type + 1)

        return ship_types

    @staticmethod
    def _generate_ships(
        island: GeoSeries,
        rng: random.Generator,
        n_ship: int,
        n_ship_type: int,
        max_ship_coast_distance: int,
        twd97: bool = True,
    ) -> GeoDataFrame:
        """
        Generates ships attacking the specified island.

        Parameters
        ----------
        island : GeoSeries
            polygon representing the island where ships are attacking
        rng : random.Generator
            random number generator
        n_ship : int
            number of ships
        n_ship_type : int
            ship type count
        max_ship_distance : int
            maximum distance between ships
        twd97 : bool, optional
            generate ships in TWD97 coordinate system, by default True

        Returns
        -------
        GeoDataFrame
            ships
        """

        # convert to TWD97 (epsg: 3826)
        island = island.to_crs(epsg=3826)

        polygon: Polygon = island.iloc[0]  # type: ignore
        radar = polygon.buffer(max_ship_coast_distance)
        min_x, min_y, max_x, max_y = radar.bounds

        ships = []
        count = 0
        while count < n_ship:
            random_point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if random_point.within(radar) and not random_point.within(polygon):
                ships.append(random_point)
                count += 1

        ships = GeoDataFrame(
            {"geometry": ships, "type": rng.integers(1, n_ship_type + 1, len(ships))},
            crs=island.crs,
        )  # type: ignore
        ships["id"] = RangeIndex(1, n_ship + 1)

        if not twd97:
            ships.to_crs(epsg=4326, inplace=True)

        return ships

    @staticmethod
    def _generate_turrets(
        island: GeoSeries,
        ships: GeoDataFrame,
        missle_types: pandas.DataFrame,
        n_turret: int,
        n_missle: int,
        rng: random.Generator,
        max_turret_coast_distance: int,
        twd97: bool = True,
    ) -> GeoDataFrame:
        """
        Generates turrets in the specified island.

        Parameters
        ----------
        island : GeoSeries
            polygon representing the island where turrets are generated
        ships : GeoDataFrame
            ships
        missle_types : pandas.DataFrame
            missle types
        n_turret : int
            number of turrets
        n_missle : int
            number of total missles
        rng : random.Generator
            random number generator
        max_turret_coast_distance : int
            maximum distance between turrets and coast
        twd97 : bool, optional
            generate turrets in TWD97 coordinate system, by default True

        Returns
        -------
        GeoDataFrame
            turrets
        """

        # convert to TWD97 (epsg: 3826)
        island = island.to_crs(epsg=3826)  # type: ignore
        ships = ships.to_crs(epsg=3826)  # type: ignore

        polygon: Polygon = island.iloc[0]  # type: ignore
        turrets = []

        count = 0

        min_x, min_y, max_x, max_y = polygon.bounds

        while count < n_turret:
            random_point = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if (
                random_point.within(polygon)
                and shortest_line(random_point, polygon.boundary).length
                < max_turret_coast_distance
            ):
                turrets.append({"geometry": random_point})
                count += 1

        # adjustable missle count
        for _ in range(n_missle):
            missle_type = rng.choice(missle_types.type)
            missle_turret = rng.choice(turrets)
            if f"missle_{missle_type}_count" not in missle_turret:
                missle_turret[f"missle_{missle_type}_count"] = 1
            else:
                missle_turret[f"missle_{missle_type}_count"] += 1

        for turret in turrets:
            for missle_type in missle_types["type"]:
                if f"missle_{missle_type}_count" not in turret:
                    turret[f"missle_{missle_type}_count"] = 0

        turrets = GeoDataFrame(turrets, crs=island.crs)
        turrets["id"] = RangeIndex(1, n_turret + 1)

        if not twd97:
            turrets.to_crs(epsg=4326, inplace=True)

        return turrets

    @classmethod
    def load_dir(cls: type[Self], input_dir: str = ".") -> Self:
        """
        Load dataset from input directory.

        Parameters
        ----------
        cls : type[Self]
            dataset class
        output_dir : str, optional
            input directory, by default "."

        Returns
        -------
        Self
            the loaded dataset
        """

        input_path = pathlib.Path(input_dir)
        if not input_path.is_dir():
            msg = f"input path {input_dir} is not a directory"
            raise FileNotFoundError(msg)

        ships = pandas.read_csv(input_path / "ships.csv", dtype={"type": numpy.int32})
        turrets = pandas.read_csv(input_path / "turrets.csv")
        missle_types = pandas.read_csv(input_path / "missles.csv")
        require_missles = pandas.read_csv(
            input_path / "ship_types.csv", dtype=numpy.int32
        )

        return cls(
            ships=ships,
            turrets=turrets,
            missle_types=missle_types,
            require_missles=require_missles,
        )

    @classmethod
    def generate(
        cls: type[Self],
        config: DatasetConfig,
        output_dir: str | None = None,
        preview: bool = False,
    ) -> Self:
        """
        Generate a dataset.

        Parameters
        ----------
        cls : type[Self]
            dataset class
        output_dir : str | None, optional
            dataset output directory, by default None
        preview : bool, optional
            generate a preview image, by default False

        Returns
        -------
        Self
            the generated dataset
        """

        sig, args = inspect.signature(cls.generate), locals()
        args = {param.name: args[param.name] for param in sig.parameters.values()}

        rng = random.Generator(random.PCG64DXSM(config.seed))

        missle_types = Dataset._generate_missle_types(
            rng=rng, n_missle_type=config.n_missle_type
        )
        require_missles = Dataset._generate_ship_types(
            rng=rng, n_ship_type=config.n_ship_type, n_missle_type=config.n_missle_type
        )

        taiwan: GeoSeries = Dataset._get_taiwan_polygon()
        ships = Dataset._generate_ships(
            island=taiwan,
            rng=rng,
            n_ship=config.n_ship,
            max_ship_coast_distance=config.max_ship_coast_distance,
            n_ship_type=config.n_ship_type,
            twd97=config.twd97,
        )
        turrets = Dataset._generate_turrets(
            island=taiwan,
            ships=ships,
            missle_types=missle_types,
            rng=rng,
            n_turret=config.n_turret,
            n_missle=config.n_missle,
            max_turret_coast_distance=config.max_turret_coast_distance,
            twd97=config.twd97,
        )
        turrets = geometry_to_xy(turrets)
        ships = geometry_to_xy(ships)

        dataset = cls(
            ships=ships,
            turrets=turrets,
            missle_types=missle_types,
            require_missles=require_missles,
        )

        if output_dir is not None:
            output_path = pathlib.Path(output_dir)
            os.makedirs(output_dir, mode=0o755, exist_ok=True)

            if preview:
                taiwan = taiwan.to_crs(epsg=3826)  # type: ignore
                plot_polygon(taiwan.geometry[0], add_points=False)  # type: ignore
                plot_points(turrets.geometry, color="blue", marker=".")  # type: ignore
                plot_points(ships.geometry, color="red", marker=".")  # type: ignore
                pyplot.savefig(output_path / "preview.png")

            with open(output_path / "args.json", "w", encoding="utf-8") as f:
                json.dump(args, f, indent=2, ensure_ascii=False)

        turrets.drop("geometry", axis=1, inplace=True)
        ships.drop("geometry", axis=1, inplace=True)

        if output_dir is not None:
            output_path = pathlib.Path(output_dir)

            columns = list(turrets.columns)
            for col in ["id", "x", "y"]:
                columns.remove(col)
            columns = ["id", "x", "y", *sorted(columns)]
            turrets[columns].to_csv(output_path / "turrets.csv", index=False)

            columns = list(ships.columns)
            columns.remove("id")
            columns = ["id", *columns]
            ships[columns].to_csv(output_path / "ships.csv", index=False)

            columns = list(missle_types.columns)
            columns.remove("type")
            columns = ["type", *columns]
            missle_types[columns].to_csv(output_path / "missles.csv", index=False)

            columns = list(require_missles.columns)
            columns.remove("type")
            columns = ["type", *sorted(columns)]
            require_missles[columns].to_csv(output_path / "ship_types.csv", index=False)

        return dataset

    def __init__(
        self,
        ships: pandas.DataFrame,
        turrets: pandas.DataFrame,
        missle_types: pandas.DataFrame,
        require_missles: pandas.DataFrame,
    ):
        self.missle_types = missle_types
        self.require_missles = require_missles.to_numpy()

        self.turrets_xy = turrets[["x", "y"]].to_numpy()
        self.turrets_missle_count = turrets.drop(columns=["x", "y"]).to_numpy()

        self.ships_xy: numpy.typing.NDArray[numpy.float64] = ships[
            ["x", "y"]
        ].to_numpy()
        self.ship_types: numpy.typing.NDArray[numpy.int64] = ships["type"].to_numpy()

        self.distances = numpy.sqrt(
            numpy.sum(
                (self.turrets_xy[:, numpy.newaxis] - self.ships_xy[numpy.newaxis, :])
                ** 2,
                axis=2,
            )
        )
        assert self.distances.shape == (
            self.turrets_xy.shape[0],
            self.ships_xy.shape[0],
        )

        self.in_range = numpy.stack([
            numpy.logical_and(
                missle_type.min_range <= self.distances,
                self.distances <= missle_type.max_range,
            )
            for missle_type in self.missle_types.itertuples()
        ])

        self.missle_damages = 1 / self.require_missles
        self.missle_damages = numpy.clip(self.missle_damages, 0, 1)

        self.missles: list[Missle] = []

        missle_id = 1
        for turret in turrets.itertuples():
            missle_types.apply(
                lambda mtype, turret=turret: self.missles.extend(
                    Missle(
                        type=int(mtype.type),  # type: ignore
                        turret_id=int(turret.id),  # type: ignore
                        id=missle_id,
                        available_targets=(
                            numpy.flatnonzero(
                                self.in_range[
                                    mtype.type - 1, turret.id - 1, :  # type: ignore
                                ]
                            )
                            + 1
                        ).tolist(),
                    )
                    for _ in range(getattr(turret, f"missle_{mtype['type']}_count"))
                ),
                axis=1,
            )

    def can_hit_ship(self, missle: Missle, ship_id: int) -> bool:
        """
        Checks if a missle can hit a ship in the dataset.

        Parameters
        ----------
        missle : Missle

        ship_id : int
            ship id

        Returns
        -------
        bool
            True if the missle can hit the ship
        """

        return (
            ship_id > 0
            and self.in_range[missle.type - 1, missle.turret_id - 1, ship_id - 1]
            and self.require_missles[self.ship_types[ship_id - 1] - 1, missle.type - 1]
            != -1
        )


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
        help="view generated dataset on map",
        action="store_true",
        default=False,
    )
    arg_parser.add_argument(
        "--twd97",
        help="use TWD97 coordinate system",
        action="store_true",
        default=True,
    )

    arg_parser.add_argument(
        "--max-turret-coast-distance",
        help="max turret-coast distance in meters",
        type=int,
        default=3000,
    )
    arg_parser.add_argument(
        "--n-turret",
        help="number of turrets",
        type=int,
        default=20,
    )
    arg_parser.add_argument(
        "--n-missle",
        help="number of missles",
        type=int,
        default=100,
    )

    arg_parser.add_argument(
        "-n",
        "--n-missle-type",
        help="number of missle types",
        type=int,
        default=2,
    )

    arg_parser.add_argument(
        "-m",
        "--n-ship-type",
        help="number of battleship types",
        type=int,
        default=10,
    )
    arg_parser.add_argument(
        "--n-ship",
        help="number of battleships",
        type=int,
        default=20,
    )
    arg_parser.add_argument(
        "--max-ship-coast-distance",
        help="maximum battleship distance to coast in meters",
        type=int,
        default=50000,
    )

    args = arg_parser.parse_args().__dict__
    logging.basicConfig(stream=sys.stderr, level=args["log_level"])
    del args["log_level"]
    Dataset.generate(**args)
