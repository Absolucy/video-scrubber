use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use opencv::{core::Mat, imgcodecs};
use std::path::Path;
use walkdir::WalkDir;

pub fn load_image(path: &Path, flags: i32) -> Result<Mat> {
	let path = path
		.to_str()
		.wrap_err("invalid path cannot be represented as a str")?;
	let img = imgcodecs::imread(path, flags)
		.wrap_err_with(|| format!("failed to read image from {}", path))?;
	crate::fixup::fixup_frame_2(&img, false)
		.wrap_err_with(|| format!("failed to fixup image from {}", path))
}

pub fn load(path: &Path, flags: i32) -> Result<Vec<Mat>> {
	if path.is_file() {
		return load_image(path, flags).map(|template| vec![template]);
	}
	let mut templates = Vec::new();
	for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
		let entry_path = entry.path();
		if !entry_path.is_file()
			|| !imgcodecs::have_image_reader(entry_path.to_str().unwrap_or_default())
				.unwrap_or(false)
		{
			continue;
		}
		let image = load_image(entry_path, flags)
			.wrap_err_with(|| format!("failed to load image at {}", entry_path.display()))?;
		templates.push(image);
	}
	Ok(templates)
}

pub fn load_multi<P: AsRef<Path>>(paths: &[P], flags: i32) -> Result<Vec<Mat>> {
	Ok(paths
		.iter()
		.map(|path| {
			let path = path.as_ref();
			load(path, flags)
				.wrap_err_with(|| format!("failed to load template(s) from {}", path.display()))
		})
		.collect::<Result<Vec<_>>>()
		.wrap_err("failed to parse templates")?
		.into_iter()
		.flatten()
		.collect::<Vec<_>>())
}
