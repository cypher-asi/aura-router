//! S3 upload service for generated images.

use aws_sdk_s3::Client as S3Client;
use base64::Engine;

/// S3 configuration.
#[derive(Clone)]
pub struct S3Config {
    pub client: S3Client,
    pub bucket: String,
    pub region: String,
}

impl S3Config {
    /// Create S3 config from environment.
    pub async fn from_env() -> Option<Self> {
        let bucket = std::env::var("S3_BUCKET_NAME").ok()?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        let config = aws_config::from_env()
            .region(aws_config::Region::new(region.clone()))
            .load()
            .await;

        let client = S3Client::new(&config);

        Some(Self {
            client,
            bucket,
            region,
        })
    }

    /// Generate an S3 key for an uploaded asset.
    fn generate_key(user_id: &str, asset_type: &str, extension: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp_millis();
        let random = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        format!("{asset_type}s/{user_id}/{timestamp}-{random}.{extension}")
    }

    /// Get the public URL for an S3 key.
    fn public_url(&self, key: &str) -> String {
        format!(
            "https://{}.s3.{}.amazonaws.com/{key}",
            self.bucket, self.region
        )
    }

    /// Detect content type and extension from base64 data URL prefix.
    fn detect_image_type(data_url: &str) -> (&'static str, &'static str) {
        if data_url.contains("image/jpeg") || data_url.contains("image/jpg") {
            ("image/jpeg", "jpg")
        } else if data_url.contains("image/webp") {
            ("image/webp", "webp")
        } else if data_url.contains("image/gif") {
            ("image/gif", "gif")
        } else {
            ("image/png", "png")
        }
    }

    /// Upload a base64-encoded image to S3.
    /// Returns the public URL.
    pub async fn upload_base64(
        &self,
        data_url: &str,
        user_id: &str,
    ) -> Result<String, String> {
        let (content_type, extension) = Self::detect_image_type(data_url);

        // Strip data URL prefix
        let base64_data = data_url
            .find(",")
            .map(|i| &data_url[i + 1..])
            .unwrap_or(data_url);

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(base64_data)
            .map_err(|e| format!("Invalid base64: {e}"))?;

        let key = Self::generate_key(user_id, "image", extension);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(bytes.into())
            .content_type(content_type)
            .metadata("userId", user_id)
            .metadata("uploadedAt", &chrono::Utc::now().to_rfc3339())
            .send()
            .await
            .map_err(|e| format!("S3 upload failed: {e}"))?;

        Ok(self.public_url(&key))
    }

    /// Upload raw bytes to S3.
    /// Returns the public URL.
    pub async fn upload_bytes(
        &self,
        bytes: Vec<u8>,
        user_id: &str,
        content_type: &str,
        extension: &str,
    ) -> Result<String, String> {
        let key = Self::generate_key(user_id, "image", extension);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(bytes.into())
            .content_type(content_type)
            .metadata("userId", user_id)
            .metadata("uploadedAt", &chrono::Utc::now().to_rfc3339())
            .send()
            .await
            .map_err(|e| format!("S3 upload failed: {e}"))?;

        Ok(self.public_url(&key))
    }
}

/// Apply a watermark to an image buffer.
/// Composites the watermark in the bottom-right corner.
/// Returns the watermarked image as PNG bytes.
pub fn apply_watermark(
    image_bytes: &[u8],
    watermark_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    use image::{GenericImageView, ImageReader};
    use std::io::Cursor;

    let main_img = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| format!("Failed to read image: {e}"))?
        .decode()
        .map_err(|e| format!("Failed to decode image: {e}"))?;

    let watermark = ImageReader::new(Cursor::new(watermark_bytes))
        .with_guessed_format()
        .map_err(|e| format!("Failed to read watermark: {e}"))?
        .decode()
        .map_err(|e| format!("Failed to decode watermark: {e}"))?;

    let (main_w, main_h) = main_img.dimensions();

    // Watermark sizing: 17.6% of main image width, max 352px
    let watermark_max_w = ((main_w as f64) * 0.176).min(352.0) as u32;
    let watermark_resized = watermark.resize(
        watermark_max_w,
        watermark_max_w, // will maintain aspect ratio
        image::imageops::FilterType::Lanczos3,
    );

    let (wm_w, wm_h) = watermark_resized.dimensions();

    // Position: bottom-right with 3% padding
    let padding = ((main_w as f64) * 0.03) as u32;
    let x = main_w.saturating_sub(wm_w + padding);
    let y = main_h.saturating_sub(wm_h + padding);

    let mut output = main_img.to_rgba8();
    image::imageops::overlay(&mut output, &watermark_resized.to_rgba8(), x as i64, y as i64);

    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    output
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode watermarked image: {e}"))?;

    Ok(buf)
}
