mod integer_composition;

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Error, Result, anyhow};
use clap::Parser;
use csv::ReaderBuilder;
use integer_composition::IntegerCompositions;
use itertools::Itertools;
use ndarray::{Array, s};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use serde::Deserialize;

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long)]
    data_dir: String,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
struct Ship {
    #[allow(dead_code)]
    pub id: u64,
    pub r#type: u64,
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Default, Clone)]
struct ShipType {
    pub r#type: u64,
    pub require_missle_count: HashMap<u64, i64>,
}

#[derive(Debug, Default, Clone)]
struct Turret {
    pub id: u64,
    pub missle_count: HashMap<u64, u64>,
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
struct MissleType {
    pub r#type: u64,
    pub min_range: u64,
    pub max_range: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let data_dir = Path::new(&args.data_dir);

    let mut reader_builder = ReaderBuilder::new();
    reader_builder.has_headers(true).delimiter(b',');

    let mut ships = reader_builder.from_path(data_dir.join("ships.csv"))?;
    let ships: Vec<Ship> = ships.deserialize().into_iter().collect::<Result<_, _>>()?;

    let mut turrets = reader_builder.from_path(data_dir.join("turrets.csv"))?;
    let turrets_header = turrets.headers()?.clone();
    let turrets_header: Vec<&str> = turrets_header.into_iter().collect();
    let turrets: Vec<Turret> = turrets
        .records()
        .into_iter()
        .map(|result| {
            let record = result?;

            let mut turret = Turret::default();
            for (key, value) in turrets_header.iter().zip(record.iter()) {
                match *key {
                    "id" => turret.id = value.parse()?,
                    "x" => turret.x = value.parse()?,
                    "y" => turret.y = value.parse()?,
                    k if k.starts_with("missle_") && k.ends_with("_count") => {
                        turret
                            .missle_count
                            .insert(k[7..=k.len() - 7].parse()?, value.parse()?);
                    }
                    _ => return Err(anyhow!("unknown key: {}", key)),
                };
            }

            Ok(turret)
        })
        .collect::<Result<_, Error>>()?;

    let mut missle_types = reader_builder.from_path(data_dir.join("missles.csv"))?;
    let missle_types: Vec<MissleType> = missle_types
        .deserialize()
        .into_iter()
        .collect::<Result<_, _>>()?;

    let mut ship_types = reader_builder.from_path(data_dir.join("ship_types.csv"))?;
    let ship_types_header = ship_types.headers()?.clone();
    let ship_types_header: Vec<&str> = ship_types_header.into_iter().collect();
    let ship_types: Vec<ShipType> = ship_types
        .records()
        .into_iter()
        .map(|result| {
            let record = result?;

            let mut ship_type = ShipType::default();
            for (key, value) in ship_types_header.iter().zip(record.iter()) {
                match *key {
                    "type" => ship_type.r#type = value.parse()?,
                    k if k.starts_with("require_missle_") && k.ends_with("_count") => {
                        ship_type
                            .require_missle_count
                            .insert(k[15..=k.len() - 7].parse()?, value.parse()?);
                    }
                    _ => return Err(anyhow!("unknown key: {}", key)),
                };
            }

            Ok(ship_type)
        })
        .collect::<Result<_, Error>>()?;

    let distances = Array::from_shape_fn((turrets.len(), ships.len()), |(i, j)| {
        let dx = turrets[i].x - ships[j].x;
        let dy = turrets[i].y - ships[j].y;

        dx * dx + dy * dy
    });
    let distances = distances.sqrt();

    let mut in_range = Vec::new();
    for missle in &missle_types {
        let missle_in_range = Array::from_shape_vec(
            [distances.shape()[0], distances.shape()[1]],
            distances
                .into_par_iter()
                .map(|d| *d <= missle.max_range as f64 && missle.min_range as f64 <= *d)
                .collect(),
        )?;
        in_range.push(missle_in_range);
    }
    let in_range = ndarray::stack(
        ndarray::Axis(0),
        in_range
            .iter()
            .map(|a| a.view())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;

    let mut missle_damages =
        Array::from_shape_fn((ship_types.len(), missle_types.len()), |(i, j)| {
            ship_types[i].require_missle_count[&missle_types[j].r#type] as f64
        });
    missle_damages.par_mapv_inplace(|v| 1.0 / v);
    let missle_damages = missle_damages.clamp(0.0, 1.0);

    let mut available_targets: Vec<Vec<Vec<_>>> =
        vec![vec![Vec::new(); missle_types.len()]; turrets.len()];
    let mut combinations_iters = VecDeque::new();
    for turret in &turrets {
        for missle_type in &missle_types {
            let count = turret.missle_count[&missle_type.r#type];
            let targets: Vec<_> = in_range
                .slice(s![
                    missle_type.r#type as usize - 1,
                    turret.id as usize - 1,
                    ..
                ])
                .iter()
                .enumerate()
                .filter_map(|(i, &is_in)| if is_in { Some(i as u64 + 1) } else { None })
                .chain(std::iter::once(0))
                .collect();
            available_targets[turret.id as usize - 1][missle_type.r#type as usize - 1] = targets;

            combinations_iters.push_back(
                IntegerCompositions::new(count as usize, available_targets.len())
                    .map(|v| (v, turret.id, missle_type.r#type)),
            );
        }
    }

    let available_targets = Arc::new(available_targets);
    let ships = Arc::new(ships);
    let missle_damages = Arc::new(missle_damages);

    let first_iter = combinations_iters.pop_front().unwrap();
    let second_iter = combinations_iters.pop_front().unwrap();
    let third_iter = combinations_iters.pop_front().unwrap();

    let reducer = |(mut best, mut best_destroyed_ships, mut best_total_damage),
                   (assignment, destroyed_ships, total_damage)| {
        if destroyed_ships > best_destroyed_ships {
            best = assignment;
            best_destroyed_ships = destroyed_ships;
            best_total_damage = total_damage;
        } else if destroyed_ships == best_destroyed_ships && total_damage > best_total_damage {
            best = assignment;
            best_total_damage = total_damage;
        }

        (best, best_destroyed_ships, best_total_damage)
    };

    let best_assignment = first_iter
        .cartesian_product(second_iter)
        .cartesian_product(third_iter)
        .par_bridge()
        .map(|((a1, a2), a3)| {
            let (a1, a2, a3) = (Arc::new(a1), Arc::new(a2), Arc::new(a3));
            combinations_iters
                .iter()
                .cloned()
                .multi_cartesian_product()
                .map(|combinations| {
                    let combination: Vec<_> = vec![a1.clone(), a2.clone(), a3.clone()]
                        .into_iter()
                        .chain(combinations.into_iter().map(|v| Arc::new(v)))
                        .collect();
                    let mut damages = vec![0.0; ships.len()];
                    for assignment in combination.iter() {
                        let (ref assignment, turret_id, missle_type) = **assignment;
                        let targets =
                            &available_targets[turret_id as usize - 1][missle_type as usize - 1];

                        // drop assignments to target 0 (not assigned)
                        let target_assignment_iter = targets
                            .iter()
                            .zip_eq(assignment.iter())
                            .take(targets.len() - 1);
                        for (target, assigned_count) in target_assignment_iter {
                            damages[*target as usize - 1] += missle_damages[[
                                ships[*target as usize - 1].r#type as usize - 1,
                                missle_type as usize - 1,
                            ]] * *assigned_count as f64;
                        }
                    }

                    let destroyed_ships = damages.iter().filter(|v| **v >= 1.0).count();
                    let total_damage = damages.iter().sum::<f64>();

                    (combination, destroyed_ships, total_damage)
                })
                .reduce(reducer)
                .unwrap()
        })
        .reduce(|| (Vec::new(), 0, 0.0), reducer);

    println!("best assignment: {:?}", best_assignment.0.iter());
    println!("destroyed ships: {}", best_assignment.1);
    println!("total damage: {}", best_assignment.2);

    Ok(())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rayon::iter::{ParallelBridge, ParallelIterator};

    #[test]
    fn test() {
        let max: i32 = (0..8)
            .map(|_| 0..=1)
            .multi_cartesian_product()
            .par_bridge()
            .map(|v| v.iter().sum())
            .max()
            .unwrap();

        assert_eq!(max, 8);
    }
}
