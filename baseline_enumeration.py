import itertools
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Thread

import numpy
import pandas


def main():
    ships = pandas.read_csv("./50W10T_0/ships.csv", dtype={"type": numpy.int32})
    turrets = pandas.read_csv("./50W10T_0/turrets.csv")
    missle_types = pandas.read_csv("./50W10T_0/missles.csv")
    require_missles = (
        pandas.read_csv("./50W10T_0/ship_types.csv", dtype=numpy.int32)
        .drop("type", axis=1)
        .to_numpy()
    )

    turrets_xy = turrets[["x", "y"]].to_numpy()
    ships_xy = ships[["x", "y"]].to_numpy()

    distances = numpy.sqrt(
        numpy.sum(
            (turrets_xy[:, numpy.newaxis] - ships_xy[numpy.newaxis, :]) ** 2, axis=2
        )
    )
    assert distances.shape == (turrets_xy.shape[0], ships_xy.shape[0])

    in_range = numpy.stack([
        numpy.logical_and(
            missle_type.min_range <= distances, distances <= missle_type.max_range
        )
        for missle_type in missle_types.itertuples()
    ])

    missle_damages = 1 / require_missles
    missle_damages = numpy.clip(missle_damages, 0, 1)

    @dataclass
    class Missle:
        id: int
        type: int
        turret_id: int

        def can_hit_ship(self, ship_id: int) -> bool:
            return (
                ship_id > 0
                and in_range[self.type - 1, self.turret_id - 1, ship_id - 1]
                and require_missles[ships.type[ship_id - 1] - 1, self.type - 1] != -1
            )

        @property
        def available_targets(self) -> list[int]:
            targets = (
                numpy.flatnonzero(in_range[self.type - 1, self.turret_id - 1, :]) + 1
            )
            targets = targets.tolist()

            assert isinstance(targets, list)
            return targets  # type: ignore

    def evaluate_destroyed_ships(
        missles: list[Missle],
        assignment: list[int],
        ships: pandas.DataFrame,
        missle_damages: numpy.typing.NDArray,
    ) -> tuple[int, float]:
        damage_on_ship = numpy.zeros(len(ships), dtype=numpy.float64)
        for i, missle in enumerate(missles):
            if missle.can_hit_ship(assignment[i]):
                damage_on_ship[assignment[i] - 1] += missle_damages[
                    ships.type[assignment[i] - 1] - 1
                ][missle.type - 1]

        damage_on_ship[damage_on_ship >= 1] = 1

        return numpy.sum(
            damage_on_ship[damage_on_ship >= 1], dtype=numpy.int64
        ).item(), numpy.sum(damage_on_ship).item()

    missles: list[Missle] = []

    missle_count = 1
    for turret in turrets.itertuples():
        for missle_type in missle_types.itertuples():
            for _ in range(getattr(turret, f"missle_{missle_type.type}_count")):
                missles.append(
                    Missle(
                        type=int(missle_type.type),  # type: ignore
                        turret_id=int(turret.id),  # type: ignore
                        id=missle_count,
                    )
                )
                missle_count += 1

    available_targets = [[*missle.available_targets, 0] for missle in missles]

    # multithread this
    queue = Queue(1024)
    best_assignment = None
    best_destroyed_ships = 0

    def produce_assignment():
        for assignment in itertools.product(*available_targets):
            queue.put(assignment)

    def evaluate_assignment(assignment):
        nonlocal best_destroyed_ships, best_assignment
        destroyed_ships, _ = evaluate_destroyed_ships(
            missles, list(assignment), ships, missle_damages
        )
        if destroyed_ships > best_destroyed_ships:
            print("New best destroyed_ships:", destroyed_ships)
            best_destroyed_ships = destroyed_ships
            best_assignment = list(assignment)

    producer = Thread(target=produce_assignment)
    producer.start()

    with ThreadPool(20):
        while True:
            assignment = queue.get()
            evaluate_assignment(assignment)
            queue.task_done()

            if best_destroyed_ships == len(ships):
                break

    print(best_assignment)
    print(best_destroyed_ships)


if __name__ == "__main__":
    main()
