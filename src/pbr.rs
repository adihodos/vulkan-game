use ash::vk::{
    ComponentMapping, ComponentSwizzle, Extent3D, Format, ImageAspectFlags, ImageCreateInfo,
    ImageLayout, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags,
    ImageViewCreateInfo, ImageViewType, MemoryPropertyFlags, SampleCountFlags, SharingMode,
};
use log::error;
use nalgebra_glm::Vec4;
use smallvec::SmallVec;

use crate::vk_renderer::{
    ImageCopySource, RendererWorkPackage, UniqueImage, UniqueImageView, VulkanRenderer,
};

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct PbrMaterial {
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texarray_id: u32,
    pub metallic_rough_texarray_id: u32,
    pub normal_texarray_id: u32,
}

/// All textures of a material that are used for PBR
pub struct PbrMaterialTextureCollection {
    pub base_color_tex: UniqueImage,
    pub base_color_imageview: UniqueImageView,
    pub metallic_roughness_tex: UniqueImage,
    pub metallic_imageview: UniqueImageView,
    pub normal_tex: UniqueImage,
    pub normal_imageview: UniqueImageView,
}

impl PbrMaterialTextureCollection {
    pub fn create(
        renderer: &VulkanRenderer,
        base_color_images: (u32, u32, Vec<ImageCopySource>),
        metallic_roughness_images: (u32, u32, Vec<ImageCopySource>),
        normal_images: (u32, u32, Vec<ImageCopySource>),
        gpu_work_pkg: &RendererWorkPackage,
    ) -> Option<PbrMaterialTextureCollection> {
        let mut images_and_views = [base_color_images, metallic_roughness_images, normal_images]
            .iter()
            .zip(
                [
                    Format::R8G8B8A8_SRGB,
                    Format::R8G8B8A8_UNORM,
                    Format::R8G8B8A8_UNORM,
                ]
                .iter(),
            )
            .filter_map(|(image_data, &image_format)| {
                let (width, height, pixels) = image_data;
                let image = UniqueImage::with_data(
                    renderer,
                    &ImageCreateInfo::builder()
                        .format(image_format)
                        .usage(ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST)
                        .tiling(ImageTiling::OPTIMAL)
                        .initial_layout(ImageLayout::UNDEFINED)
                        .array_layers(pixels.len() as u32)
                        .mip_levels(1)
                        .samples(SampleCountFlags::TYPE_1)
                        .sharing_mode(SharingMode::EXCLUSIVE)
                        .image_type(ImageType::TYPE_2D)
                        .extent(
                            Extent3D::builder()
                                .width(*width)
                                .height(*height)
                                .depth(1)
                                .build(),
                        )
                        .build(),
                    &pixels,
                    &gpu_work_pkg,
                )?;

                let imageview = UniqueImageView::new(
                    renderer.graphics_device(),
                    &ImageViewCreateInfo::builder()
                        .format(image_format)
                        .view_type(ImageViewType::TYPE_2D_ARRAY)
                        .image(image.image)
                        .components(
                            ComponentMapping::builder()
                                .r(ComponentSwizzle::IDENTITY)
                                .g(ComponentSwizzle::IDENTITY)
                                .b(ComponentSwizzle::IDENTITY)
                                .a(ComponentSwizzle::IDENTITY)
                                .build(),
                        )
                        .subresource_range(
                            ImageSubresourceRange::builder()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .base_array_layer(0)
                                .base_mip_level(0)
                                .layer_count(pixels.len() as u32)
                                .level_count(1)
                                .build(),
                        )
                        .build(),
                )?;
                Some((image, imageview))
            })
            .collect::<SmallVec<[(UniqueImage, UniqueImageView); 4]>>();

        if images_and_views.len() != 3 {
            error!("Error creating PBR texture");
            return None;
        }

        let (normal_tex, normal_imageview) = images_and_views.swap_remove(2);
        let (metallic_roughness_tex, metallic_imageview) = images_and_views.swap_remove(1);
        let (base_color_tex, base_color_imageview) = images_and_views.swap_remove(0);

        Some(PbrMaterialTextureCollection {
            base_color_tex,
            base_color_imageview,
            metallic_roughness_tex,
            metallic_imageview,
            normal_tex,
            normal_imageview,
        })
    }
}
