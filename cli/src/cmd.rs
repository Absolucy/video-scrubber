use clap::{Args, Parser, Subcommand};
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use itertools::Itertools;
use std::{path::PathBuf, str::FromStr};
use video_scrubber_core::opencv::core::Rect;

#[derive(Parser)]
#[command(author, version, about, long_about = None, propagate_version = true)]
pub struct CliArgs {
	#[command(subcommand)]
	pub command: CliSubcommands,
}

#[derive(Subcommand)]
pub enum CliSubcommands {
	Scrub(ScrubArgs),
	Test(TestArgs),
}

#[derive(Args)]
pub struct ScrubArgs {
	/// The input video file.
	#[arg(short, long)]
	pub input: PathBuf,
	/// The template image file.
	#[arg(short = 'p',  required = true, num_args = 1..)]
	pub pos_templates: Vec<PathBuf>,
	/// The negative template image files.
	#[arg(short = 'n', allow_hyphen_values = true)]
	pub neg_templates: Vec<PathBuf>,
	/// The file to output to.
	#[arg(short, default_value = "output.mkv")]
	pub output: PathBuf,
	/// The minimum match threshold (0-1).
	#[arg(short = 'm', default_value = "0.7")]
	pub pos_threshold: f64,
	/// The minimum negative match threshold (0-1).
	#[arg(short = 'x', default_value = "0.7")]
	pub neg_threshold: f64,
	/// How many seconds to pad out removal ranges with, just to be sure.
	#[arg(short = 'f', long, default_value = "1.0")]
	pub padding: f64,
	/// The bounds of the region of interest (x,y,width,height).
	#[arg(short, long, value_parser = parse_rect)]
	pub bounds: Option<Rect>,
	/// How many threads to use. Defaults to the amount of logical cores.
	#[arg(short = 'j', long)]
	pub threads: Option<usize>,
	/// The ffmpeg options to force.
	#[arg(short = 'o', long)]
	pub ffmpeg_opts: Option<String>,
}

#[derive(Args)]
#[command(author, version, about, long_about = None)]
pub struct TestArgs {
	/// The input image files.
	#[arg(short, long, required = true, num_args = 1..)]
	pub input: Vec<PathBuf>,
	/// The template image file.
	#[arg(short, long, required = true, num_args = 1..)]
	pub template: Vec<PathBuf>,
	/// The negative template image files.
	#[arg(short, long)]
	pub negative_template: Vec<PathBuf>,
	/// The minimum match threshold (0-1).
	#[arg(short = 'm', long, default_value = "0.7")]
	pub threshold: f64,
	/// The minimum negative match threshold (0-1).
	#[arg(short = 'x', long, default_value = "0.6")]
	pub negative_threshold: f64,
	/// The bounds of the region of interest (x,y,width,height).
	#[arg(short, long, value_parser = parse_rect)]
	pub bounds: Option<Rect>,
	#[arg(long)]
	pub color: bool,
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
