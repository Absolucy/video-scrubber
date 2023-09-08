use crate::{cmd::TestArgs, frame};
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use opencv::{
	core::{Mat, Size, BORDER_DEFAULT, NORM_MINMAX},
	imgcodecs::{self, IMREAD_COLOR, IMREAD_GRAYSCALE},
	imgproc,
};
use std::path::Path;

fn read_image(path: &Path, mode: i32) -> Result<Mat> {
	let image = imgcodecs::imread(
		path.to_str()
			.wrap_err("invalid path cannot be represented as a str")?,
		mode,
	)
	.wrap_err_with(|| format!("failed to read image from {}", path.display()))?;
	let mut blurred = Mat::default();
	imgproc::gaussian_blur(
		&image,
		&mut blurred,
		Size::new(5, 5),
		0.0,
		0.0,
		BORDER_DEFAULT,
	)
	.wrap_err("failed to apply gaussian blur")?;
	let mut normalized = Mat::default();
	opencv::core::normalize(
		&blurred,
		&mut normalized,
		0.0,
		255.0,
		NORM_MINMAX,
		-1,
		&Mat::default(),
	)
	.wrap_err("failed to normalize image")?;
	Ok(normalized)
}

pub fn test(args: TestArgs) -> Result<()> {
	let mode = if args.color {
		IMREAD_COLOR
	} else {
		IMREAD_GRAYSCALE
	};
	let pos_templates = args
		.template
		.iter()
		.map(|path| {
			read_image(path, mode)
				.wrap_err("failed to read template")
				.map(|img| (img, path.to_owned()))
		})
		.collect::<Result<Vec<_>>>()
		.wrap_err("failed to read templates")?;
	let neg_templates = args
		.negative_template
		.iter()
		.map(|path| read_image(path, mode).wrap_err("failed to read negative template"))
		.collect::<Result<Vec<_>>>()
		.wrap_err("failed to read negative templates")?;

	for input in args.input {
		let frame = read_image(&input, mode)
			.wrap_err_with(|| format!("failed to read frame image from {}", input.display()))?;

		let mut result = Mat::default();
		let frame = match args.bounds {
			Some(bounds) => Mat::roi(&frame, bounds).wrap_err("invalid roi")?,
			None => frame,
		};
		for (template, path) in pos_templates.clone() {
			let (matched, pos, neg) = frame::cpu::process_frame(
				None,
				&mut result,
				frame.clone(),
				&[template],
				&neg_templates,
				None,
				None,
			)
			.wrap_err("failed to get matches")?;

			println!(
				"[{}] {}: matched={}, pos={:.2}, neg={:.2}",
				input.file_name().unwrap().to_string_lossy(),
				path.display(),
				matched,
				pos,
				neg
			);
		}
	}

	Ok(())
}
