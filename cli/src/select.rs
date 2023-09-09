use crate::cmd::SelectArgs;
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use video_scrubber_core::opencv::{
	highgui,
	imgcodecs::{self, IMREAD_COLOR},
};

pub fn select(args: SelectArgs) -> Result<()> {
	let path = args
		.input
		.to_str()
		.wrap_err("invalid path cannot be represented as a str")?;
	if !imgcodecs::have_image_reader(path)? {
		panic!("cannot read image at {path}, opencv does not support this format");
	}
	let img = imgcodecs::imread(path, IMREAD_COLOR)
		.wrap_err_with(|| format!("failed to read image from {path}"))?;
	let roi = highgui::select_roi("Select ROI", &img, true, false, true)
		.wrap_err("failed to select roi")?;
	println!("{},{},{},{}", roi.x, roi.y, roi.width, roi.height);
	Ok(())
}
