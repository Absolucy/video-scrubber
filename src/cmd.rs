use clap::Parser;
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use itertools::Itertools;
use opencv::core::Rect;
use std::{path::PathBuf, str::FromStr};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
	/// The input video file.
	#[arg(short, long)]
	pub input: PathBuf,
	/// The template image file.
	#[arg(short, long, default_value = "template.png")]
	pub template: PathBuf,
	/// The file to output to.
	#[arg(short, default_value = "output.mkv")]
	pub output: PathBuf,
	/// The minimum match threshold (0-1).
	#[arg(short = 'm', long, default_value = "0.7")]
	pub threshold: f64,
	/// How many seconds to pad out removal ranges with, just to be sure.
	#[arg(short = 'p', long, default_value = "1.0")]
	pub padding: f64,
	/// The bounds of the region of interest (x,y,width,height).
	#[arg(short, long, value_parser = parse_rect)]
	pub bounds: Option<Rect>,
	/// Use CUDA acceleration.
	#[arg(long)]
	pub cuda: bool,
	/// How many threads to use. Defaults to the amount of logical cores.
	#[arg(short = 'j', long)]
	pub threads: Option<usize>,
}

fn parse_rect(arg: &str) -> Result<Rect> {
	let (x, y, width, height) = arg
		.split(',')
		.map(str::trim)
		.map(|thingy| {
			i32::from_str(thingy).wrap_err_with(|| format!("invalid number '{}'", thingy))
		})
		.collect::<Result<Vec<i32>>>()
		.wrap_err("rectangle should be formatted at x,y,width,height")?
		.into_iter()
		.collect_tuple()
		.context("rectangle should be formatted at x,y,width,height")?;

	Ok(Rect::new(x, y, width, height))
}
