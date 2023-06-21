#![allow(dead_code)]

use log::{error, info, warn};
use smallvec::SmallVec;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{c_char, c_void, CStr, CString},
    mem::size_of,
    path::PathBuf,
    ptr::copy_nonoverlapping,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    prelude::VkResult,
    vk::{
        self, AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference,
        AttachmentReferenceBuilder, AttachmentStoreOp, BlendFactor, BlendOp, Bool32, Buffer,
        BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsageFlags, ClearColorValue,
        ClearDepthStencilValue, ClearValue, ColorComponentFlags, CommandBuffer,
        CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferResetFlags, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags,
        CommandPoolCreateInfo, CompareOp, ComponentMapping, ComponentSwizzle,
        CompositeAlphaFlagsKHR, CullModeFlags, DebugUtilsMessageSeverityFlagsEXT,
        DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCallbackDataEXT,
        DebugUtilsMessengerCreateFlagsEXT, DebugUtilsMessengerCreateInfoEXT,
        DebugUtilsMessengerEXT, DependencyFlags, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
        DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType,
        DeviceCreateInfo, DeviceMemory, DeviceQueueCreateInfo, DeviceSize, DynamicState, Extent2D,
        Extent3D, Fence, FenceCreateFlags, FenceCreateInfo, Format, FormatFeatureFlags,
        Framebuffer, FramebufferCreateInfo, FrontFace, GraphicsPipelineCreateInfo, Image,
        ImageAspectFlags, ImageCreateFlags, ImageCreateInfo, ImageLayout, ImageMemoryBarrier,
        ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageType, ImageUsageFlags,
        ImageView, ImageViewCreateInfo, ImageViewType, MappedMemoryRange, MemoryAllocateInfo,
        MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements, Offset2D, Offset3D,
        PFN_vkCreateFence, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceFeatures2,
        PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, PhysicalDeviceType,
        PhysicalDeviceVulkan11Features, Pipeline, PipelineBindPoint, PipelineCache,
        PipelineCacheCreateInfo, PipelineColorBlendAttachmentState,
        PipelineColorBlendAttachmentStateBuilder, PipelineColorBlendStateCreateInfo,
        PipelineDepthStencilStateCreateInfo, PipelineDynamicStateCreateInfo,
        PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateInfo,
        PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo,
        PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineVertexInputStateCreateInfo,
        PipelineViewportStateCreateInfo, PolygonMode, PresentInfoKHR, PresentModeKHR,
        PrimitiveTopology, PushConstantRange, Queue, QueueFlags, Rect2D, RenderPass,
        RenderPassBeginInfo, RenderPassCreateInfo, RenderingInfo, SampleCountFlags, SampleMask,
        SamplerCreateInfo, Semaphore, SemaphoreCreateInfo, ShaderModule, ShaderModuleCreateInfo,
        ShaderStageFlags, SharingMode, SubmitInfo, SubpassContents, SubpassDependency,
        SubpassDescription, SubpassDescriptionFlags, SurfaceCapabilitiesKHR, SurfaceFormatKHR,
        SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR, VertexInputAttributeDescription,
        VertexInputBindingDescription, Viewport, WriteDescriptorSet, HINSTANCE,
        QUEUE_FAMILY_IGNORED, SUBPASS_EXTERNAL, WHOLE_SIZE,
    },
    Device, Entry, Instance,
};

#[derive(Clone, Debug)]
pub struct UniqueDeviceMemory {
    pub memory: DeviceMemory,
    device: *const Device,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ImageInfo {
    pub is_array: bool,
    pub is_cubemap: bool,
    pub num_faces: u32,
    pub num_layers: u32,
    pub num_levels: u32,
    pub num_dimensions: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: Format,
    pub image_type: ImageType,
    pub view_type: ImageViewType,
}

impl std::convert::From<ktx2::Header> for ImageInfo {
    fn from(header: ktx2::Header) -> Self {
        let num_dimensions = if header.pixel_depth > 0 {
            3
        } else if header.pixel_height > 0 {
            2
        } else {
            1
        };

        let is_cubemap = header.face_count == 6;
        let is_array = header.layer_count.max(1) > 1;
        let layer_count = if is_cubemap {
            header.layer_count.max(1) * 6
        } else {
            header.layer_count.max(1)
        };

        let (image_type, view_type) =
            UniqueImage::get_image_and_view_type(num_dimensions, is_array, is_cubemap);

        let format = header
            .format
            .expect("KTX2 header does not specify a format");

        ImageInfo {
            is_array,
            is_cubemap,
            num_faces: header.face_count,
            num_layers: layer_count,
            num_levels: header.level_count,
            num_dimensions,
            width: header.pixel_width,
            height: header.pixel_height,
            depth: header.pixel_depth.max(1),
            format: Format::from_raw(format.0.get() as i32),
            image_type,
            view_type,
        }
    }
}

impl std::convert::From<ImageCreateInfo> for ImageInfo {
    fn from(img_create_info: ImageCreateInfo) -> Self {
        let is_cubemap = img_create_info
            .flags
            .contains(ImageCreateFlags::CUBE_COMPATIBLE);

        let is_array = if is_cubemap {
            (img_create_info.array_layers / 6) > 1
        } else {
            img_create_info.array_layers > 1
        };

        let num_dimensions = UniqueImage::get_dimensions_from_type(img_create_info.image_type);
        let (_, view_type) =
            UniqueImage::get_image_and_view_type(num_dimensions, is_array, is_cubemap);

        ImageInfo {
            is_array,
            is_cubemap,
            num_faces: if is_cubemap { 6 } else { 1 },
            num_layers: img_create_info.array_layers,
            num_levels: img_create_info.mip_levels,
            num_dimensions,
            width: img_create_info.extent.width,
            height: img_create_info.extent.height,
            depth: img_create_info.extent.depth,
            format: img_create_info.format,
            image_type: img_create_info.image_type,
            view_type,
        }
    }
}

impl std::ops::Drop for UniqueDeviceMemory {
    fn drop(&mut self) {
        unsafe { (*self.device).free_memory(self.memory, None) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ImageCopySource {
    pub src: *const u8,
    pub bytes: DeviceSize,
}

pub struct UniqueImage {
    pub image: Image,
    pub memory: DeviceMemory,
    pub info: ImageInfo,
    device: *const Device,
}

impl UniqueImage {
    pub fn new(
        graphics_device: &Device,
        memory_props: &PhysicalDeviceMemoryProperties,
        image_info: &ImageCreateInfo,
    ) -> Option<UniqueImage> {
        let image = unsafe { graphics_device.create_image(image_info, None) }
            .map_err(|e| error!("Failed to create image: {}", e))
            .ok()?;

        let image_memory_req = unsafe { graphics_device.get_image_memory_requirements(image) };

        let memory = unsafe {
            graphics_device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(image_memory_req.size as DeviceSize)
                    .memory_type_index(choose_memory_type(
                        memory_props,
                        &image_memory_req,
                        MemoryPropertyFlags::DEVICE_LOCAL,
                    ))
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create image memory: {}", e))
        .ok()?;

        unsafe {
            let _ = graphics_device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| error!("Failed to bind device memory for image, error {}", e))
                .ok()?;
        }

        Some(UniqueImage {
            image,
            memory,
            device: graphics_device as *const _,
            info: (*image_info).into(),
        })
    }

    fn set_image_layout(
        graphics_device: &Device,
        cmd_buffer: CommandBuffer,
        image: Image,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        subresource_range: ImageSubresourceRange,
    ) {
        // Source layouts (old)
        // The source access mask controls actions to be finished on the old
        // layout before it will be transitioned to the new layout.
        let src_access_mask = match old_layout {
            ImageLayout::UNDEFINED =>
            // Image layout is undefined (or does not matter).
            // Only valid as initial layout. No flags required.
            {
                AccessFlags::empty()
            }

            ImageLayout::PREINITIALIZED =>
            // Image is preinitialized.
            // Only valid as initial layout for linear images; preserves memory
            // contents. Make sure host writes have finished.
            {
                AccessFlags::HOST_WRITE
            }

            ImageLayout::COLOR_ATTACHMENT_OPTIMAL =>
            // Image is a color attachment.
            // Make sure writes to the color buffer have finished
            {
                AccessFlags::COLOR_ATTACHMENT_WRITE
            }

            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL =>
            // Image is a depth/stencil attachment.
            // Make sure any writes to the depth/stencil buffer have finished.
            {
                AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }

            ImageLayout::TRANSFER_SRC_OPTIMAL =>
            // Image is a transfer source.
            // Make sure any reads from the image have finished
            {
                AccessFlags::TRANSFER_READ
            }

            ImageLayout::TRANSFER_DST_OPTIMAL =>
            // Image is a transfer destination.
            // Make sure any writes to the image have finished.
            {
                AccessFlags::TRANSFER_WRITE
            }

            ImageLayout::SHADER_READ_ONLY_OPTIMAL =>
            // Image is read by a shader.
            // Make sure any shader reads from the image have finished
            {
                AccessFlags::SHADER_READ
            }

            _ => panic!("Unsupported layout {:?}", old_layout), /* Value not used by callers, so not supported. */
        };

        // Target layouts (new)
        // The destination access mask controls the dependency for the new image
        // layout.
        let (src_access_mask, dst_access_mask) = match new_layout {
            ImageLayout::TRANSFER_DST_OPTIMAL =>
            // Image will be used as a transfer destination.
            // Make sure any writes to the image have finished.
            {
                (src_access_mask, AccessFlags::TRANSFER_WRITE)
            }

            ImageLayout::TRANSFER_SRC_OPTIMAL =>
            // Image will be used as a transfer source.
            // Make sure any reads from and writes to the image have finished.
            {
                (
                    src_access_mask | AccessFlags::TRANSFER_READ,
                    AccessFlags::TRANSFER_READ,
                )
            }

            ImageLayout::COLOR_ATTACHMENT_OPTIMAL =>
            // Image will be used as a color attachment.
            // Make sure any writes to the color buffer have finished.
            {
                (
                    AccessFlags::TRANSFER_READ,
                    AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
            }

            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL =>
            // Image layout will be used as a depth/stencil attachment.
            // Make sure any writes to depth/stencil buffer have finished.
            {
                (src_access_mask, AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            }

            ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
                // Image will be read in a shader (sampler, input attachment).
                // Make sure any writes to the image have finished.
                let src_acc_msk = if src_access_mask.is_empty() {
                    AccessFlags::HOST_WRITE | AccessFlags::TRANSFER_WRITE
                } else {
                    src_access_mask
                };

                (src_acc_msk, AccessFlags::SHADER_READ)
            }

            _ => panic!("Unsupported layout {:?}", new_layout),
        };
        let image_memory_barrier = [ImageMemoryBarrier::builder()
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .build()];

        unsafe {
            graphics_device.cmd_pipeline_barrier(
                cmd_buffer,
                PipelineStageFlags::ALL_COMMANDS,
                PipelineStageFlags::ALL_COMMANDS,
                DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barrier,
            );
        }
    }

    pub fn with_data(
        renderer: &VulkanRenderer,
        image_info: &ImageCreateInfo,
        data: &[ImageCopySource],
        work_package: &RendererWorkPackage,
    ) -> Option<UniqueImage> {
        let graphics_device = renderer.graphics_device();

        let image = Self::new(
            renderer.graphics_device(),
            renderer.device_memory(),
            &image_info,
        )?;

        let image_memory_req =
            unsafe { graphics_device.get_image_memory_requirements(image.image) };

        let staging_buffer = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE,
            image_memory_req.size,
        )?;

        ScopedBufferMapping::create(renderer, &staging_buffer, image_memory_req.size, 0).map(
            |mapping| {
                data.iter().fold(0isize, |dst_offset, copy_src| {
                    unsafe {
                        copy_nonoverlapping(
                            copy_src.src,
                            (mapping.memptr as *mut u8).offset(dst_offset),
                            copy_src.bytes as usize,
                        );
                    }
                    dst_offset + copy_src.bytes as isize
                });
            },
        )?;

        //
        // transition image UNDEFINED -> TRANSFER_DST
        let subresource_range = ImageSubresourceRange::builder()
            .level_count(image_info.mip_levels)
            .layer_count(image_info.array_layers)
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .base_array_layer(0)
            .build();

        let image_mem_barriers = [
            ImageMemoryBarrier::builder()
                .old_layout(ImageLayout::UNDEFINED)
                .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(AccessFlags::NONE)
                .dst_access_mask(AccessFlags::MEMORY_WRITE)
                .image(image.image)
                .subresource_range(subresource_range)
                .build(),
            ImageMemoryBarrier::builder()
                .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(AccessFlags::MEMORY_READ)
                .dst_access_mask(AccessFlags::MEMORY_WRITE)
                .image(image.image)
                .subresource_range(subresource_range)
                .build(),
        ];

        let cmd_buf = work_package.cmd_buf;
        //renderer.res_loader().cmd_buf;

        unsafe {
            graphics_device.cmd_pipeline_barrier(
                cmd_buf,
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                DependencyFlags::empty(),
                &[],
                &[],
                &image_mem_barriers[0..1],
            );
        }

        //
        // buffer 2 image copy
        unsafe {
            let buffer_copy_regions = [BufferImageCopy::builder()
                .image_extent(image_info.extent)
                .image_subresource(
                    ImageSubresourceLayers::builder()
                        .base_array_layer(0)
                        .layer_count(image_info.array_layers)
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .build(),
                )
                .build()];

            graphics_device.cmd_copy_buffer_to_image(
                cmd_buf,
                staging_buffer.buffer,
                image.image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_copy_regions,
            );
        }

        //
        // Transition image TRANSFER_DST -> shader-readonly optimal
        unsafe {
            graphics_device.cmd_pipeline_barrier(
                cmd_buf,
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::FRAGMENT_SHADER,
                DependencyFlags::empty(),
                &[],
                &[],
                &image_mem_barriers[1..],
            );
        }

        renderer.push_staging_buffer(staging_buffer);

        Some(image)
    }

    fn get_dimensions_from_type(image_type: ImageType) -> u32 {
        match image_type {
            ImageType::TYPE_1D => 1,
            ImageType::TYPE_2D => 2,
            ImageType::TYPE_3D => 3,
            _ => panic!("Unsupported image type {:?}", image_type),
        }
    }

    fn get_image_and_view_type(
        num_dimensions: u32,
        is_array: bool,
        is_cubemap: bool,
    ) -> (ImageType, ImageViewType) {
        match num_dimensions {
            1 => {
                let image_type = ImageType::TYPE_1D;
                let view_type = if is_array {
                    ImageViewType::TYPE_1D_ARRAY
                } else {
                    ImageViewType::TYPE_1D
                };

                (image_type, view_type)
            }

            2 => {
                let image_type = ImageType::TYPE_2D;
                let view_type = if is_cubemap {
                    if is_array {
                        ImageViewType::CUBE_ARRAY
                    } else {
                        ImageViewType::CUBE
                    }
                } else {
                    if is_array {
                        ImageViewType::TYPE_2D_ARRAY
                    } else {
                        ImageViewType::TYPE_2D
                    }
                };

                (image_type, view_type)
            }

            3 => {
                if is_array {
                    panic!("3D array textures are not supported in Vulkan!");
                }

                (ImageType::TYPE_3D, ImageViewType::TYPE_3D)
            }

            _ => panic!(
                "Invalid dimensions specified in KTX2 texture {}",
                num_dimensions
            ),
        }
    }

    pub fn from_ktx<P: AsRef<std::path::Path>>(
        renderer: &VulkanRenderer,
        tiling: ImageTiling,
        usage_flags: ImageUsageFlags,
        final_layout: ImageLayout,
        work_package: &RendererWorkPackage,
        path: P,
    ) -> Option<UniqueImage> {
        let ktx_file = std::fs::File::open(path.as_ref()).expect(&format!(
            "Failed to open texture file {}",
            path.as_ref().display()
        ));
        let ktx_file_mapping = unsafe {
            mmap_rs::MmapOptions::new(ktx_file.metadata().unwrap().len() as usize)
                .with_file(ktx_file, 0)
                .map()
                .expect("Failed to mmap texture file")
        };

        let reader = ktx2::Reader::new(&ktx_file_mapping).expect("Failed to read KTX2");
        let header = reader.header();

        log::info!("KTX2 header {:?}", header);

        if header.pixel_width == 0 || (header.pixel_depth >= 1 && header.pixel_height == 0) {
            log::error!("Invalid KTX2 texture dimensions");
            return None;
        }

        let image_info = ImageInfo::from(header);
        log::info!("Image info: {:?}", image_info);

        let image_create_flags = if image_info.is_cubemap {
            ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            ImageCreateFlags::empty()
        };

        let usage_flags = if tiling == ImageTiling::OPTIMAL {
            usage_flags | ImageUsageFlags::TRANSFER_DST
        } else {
            usage_flags
        };

        let image_fmt_properties = unsafe {
            renderer
                .instance()
                .get_physical_device_image_format_properties(
                    renderer.phys_device,
                    image_info.format,
                    image_info.image_type,
                    tiling,
                    usage_flags,
                    image_create_flags,
                )
        }
        .expect("Failed to query image format properties");

        if image_fmt_properties.max_array_layers < image_info.num_layers {
            error!(
                "Texture needs  {} layers but the device only supports {})",
                image_info.num_layers, image_fmt_properties.max_array_layers
            );
            return None;
        }

        if image_info.num_levels > image_fmt_properties.max_mip_levels {
            error!("Texture specifies {} mip levels, but the device supports only {} mip levels for this format",
                  image_info.num_levels, image_fmt_properties.max_mip_levels );
            return None;
        }

        let texture_size = reader.data().len();

        let staging_buffer = UniqueBuffer::new(
            renderer,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE,
            texture_size as DeviceSize,
        )?;

        ScopedBufferMapping::create(renderer, &staging_buffer, WHOLE_SIZE, 0)
            .map(|staging_buffer_mapping| unsafe {
                copy_nonoverlapping(
                    reader.data().as_ptr(),
                    staging_buffer_mapping.memptr() as *mut u8,
                    texture_size,
                );
            })
            .expect("Failed to map staging buffer into host memory");

        let mut buffer_offset = texture_size;

        let copy_region_layer_count = header.layer_count.max(1) * header.face_count;

        let copy_regions = reader
            .levels()
            .enumerate()
            .map(|(level, pixels)| {
                let width = image_info.width >> level as u32;
                let height = image_info.height >> level as u32;
                let depth = image_info.depth;
                log::info!("Mip level {}, width {}, height {}", level, width, height);

                buffer_offset -= pixels.len() as usize;

                let copy_region = BufferImageCopy::builder()
                    .buffer_offset(buffer_offset as DeviceSize)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        ImageSubresourceLayers::builder()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .layer_count(copy_region_layer_count)
                            .mip_level(level as u32)
                            .base_array_layer(0)
                            .build(),
                    )
                    .image_offset(Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(Extent3D {
                        width: width as u32,
                        height: height as u32,
                        depth: depth as u32,
                    })
                    .build();

                copy_region
            })
            .collect::<SmallVec<[BufferImageCopy; 8]>>();

        let image_create_info = ImageCreateInfo::builder()
            .flags(image_create_flags)
            .format(image_info.format)
            .initial_layout(ImageLayout::UNDEFINED)
            .tiling(ImageTiling::OPTIMAL)
            .usage(usage_flags)
            .samples(SampleCountFlags::TYPE_1)
            .image_type(image_info.image_type)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .array_layers(image_info.num_layers)
            .mip_levels(image_info.num_levels)
            .extent(Extent3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            })
            .build();

        let image = UniqueImage::new(
            renderer.graphics_device(),
            renderer.device_memory(),
            &image_create_info,
        )?;

        let subresource_range = ImageSubresourceRange::builder()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(image_info.num_levels)
            .base_array_layer(0)
            .layer_count(image_info.num_layers)
            .build();

        //
        // transition undefined -> transfer dst
        Self::set_image_layout(
            renderer.graphics_device(),
            work_package.cmd_buf,
            image.image,
            ImageLayout::UNDEFINED,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            subresource_range,
        );

        //
        // copy from staging buffer to image
        unsafe {
            renderer.graphics_device().cmd_copy_buffer_to_image(
                work_package.cmd_buf,
                staging_buffer.buffer,
                image.image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );
        }

        renderer.push_staging_buffer(staging_buffer);

        //
        // transition transfer dst -> shader readonly
        Self::set_image_layout(
            renderer.graphics_device(),
            work_package.cmd_buf,
            image.image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            subresource_range,
        );

        Some(image)
    }
}

impl std::ops::Drop for UniqueImage {
    fn drop(&mut self) {
        unsafe {
            (*self.device).free_memory(self.memory, None);
            (*self.device).destroy_image(self.image, None);
        }
    }
}

pub struct UniqueImageView {
    pub view: ImageView,
    device: *const Device,
}

impl UniqueImageView {
    pub fn new(
        graphics_device: &Device,
        view_create_info: &ImageViewCreateInfo,
    ) -> Option<UniqueImageView> {
        let view = unsafe { graphics_device.create_image_view(view_create_info, None) }
            .map_err(|e| error!("Failed to create image view: {}", e))
            .ok()?;

        Some(UniqueImageView {
            view,
            device: graphics_device as *const _,
        })
    }

    pub fn from_image(renderer: &VulkanRenderer, image: &UniqueImage) -> Option<UniqueImageView> {
        Self::new(
            renderer.graphics_device(),
            &ImageViewCreateInfo::builder()
                .image(image.image)
                .view_type(image.info.view_type)
                .format(image.info.format)
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .layer_count(image.info.num_layers)
                        .level_count(image.info.num_levels)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .build(),
                ),
        )
    }
}

impl std::ops::Drop for UniqueImageView {
    fn drop(&mut self) {
        unsafe { (*self.device).destroy_image_view(self.view, None) }
    }
}

pub struct UniqueImageWithView(UniqueImage, UniqueImageView);

impl UniqueImageWithView {
    pub fn from_ktx<P: AsRef<std::path::Path>>(
        renderer: &VulkanRenderer,
        work_pkg: &RendererWorkPackage,
        path: P,
    ) -> Option<UniqueImageWithView> {
        let image = UniqueImage::from_ktx(
            renderer,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::SAMPLED,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            work_pkg,
            path,
        )?;

        let image_view = UniqueImageView::from_image(renderer, &image)?;

        Some(UniqueImageWithView(image, image_view))
    }

    pub fn image(&self) -> Image {
        self.0.image
    }

    pub fn image_view(&self) -> ImageView {
        self.1.view
    }

    pub fn info(&self) -> &ImageInfo {
        &self.0.info
    }
}

pub struct UniqueCommandPool {
    pub cmd_pool: CommandPool,
    device: *const Device,
}

impl UniqueCommandPool {
    pub fn new(
        graphics_device: &Device,
        cmd_pool_create_info: &CommandPoolCreateInfo,
    ) -> Option<UniqueCommandPool> {
        unsafe { graphics_device.create_command_pool(cmd_pool_create_info, None) }
            .map_err(|e| error!("Failed to create command pool: {}", e))
            .map(|cmd_pool| UniqueCommandPool {
                cmd_pool,
                device: &*graphics_device as *const _,
            })
            .ok()
    }
}

impl std::ops::Drop for UniqueCommandPool {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_command_pool(self.cmd_pool, None);
        }
    }
}

pub struct UniqueFramebuffer {
    pub framebuffer: Framebuffer,
    device: *const Device,
}

impl UniqueFramebuffer {
    pub fn new(
        graphics_device: &Device,
        framebuffer_create_info: &FramebufferCreateInfo,
    ) -> Option<UniqueFramebuffer> {
        unsafe { graphics_device.create_framebuffer(framebuffer_create_info, None) }
            .map_err(|e| error!("Failed to create framebuffer {}", e))
            .map(|fb| UniqueFramebuffer {
                framebuffer: fb,
                device: graphics_device as *const _,
            })
            .ok()
    }
}

impl std::ops::Drop for UniqueFramebuffer {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_framebuffer(self.framebuffer, None);
        }
    }
}

pub struct UniqueFence {
    pub fence: Fence,
    device: *const Device,
}

impl UniqueFence {
    pub fn new(graphics_device: &Device, signaled: bool) -> Option<UniqueFence> {
        let fence_create_info = if signaled {
            FenceCreateInfo::builder()
                .flags(FenceCreateFlags::SIGNALED)
                .build()
        } else {
            FenceCreateInfo::builder().build()
        };

        let fence = unsafe { graphics_device.create_fence(&fence_create_info, None) }
            .map_err(|e| error!("Failed to create fence {}", e))
            .ok()?;

        Some(UniqueFence {
            fence,
            device: graphics_device as *const _,
        })
    }
}

impl std::ops::Drop for UniqueFence {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_fence(self.fence, None);
        }
    }
}

pub struct UniqueSemaphore {
    pub semaphore: Semaphore,
    device: *const Device,
}

impl UniqueSemaphore {
    pub fn new(graphics_device: &Device) -> Option<UniqueSemaphore> {
        unsafe { graphics_device.create_semaphore(&SemaphoreCreateInfo::builder().build(), None) }
            .map_err(|e| error!("Failed to create semaphore: {}", e))
            .map(|semaphore| UniqueSemaphore {
                semaphore,
                device: graphics_device as *const _,
            })
            .ok()
    }
}

impl std::ops::Drop for UniqueSemaphore {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_semaphore(self.semaphore, None);
        }
    }
}

pub struct UniqueDescriptorPool {
    pub dpool: DescriptorPool,
    device: *const Device,
}

pub struct DescriptorPoolBuilder {
    pools: Vec<DescriptorPoolSize>,
}

impl DescriptorPoolBuilder {
    pub fn new() -> Self {
        Self {
            pools: Vec::with_capacity(8),
        }
    }

    pub fn add_pool(mut self, pool_type: DescriptorType, count: u32) -> Self {
        self.pools.push(
            DescriptorPoolSize::builder()
                .ty(pool_type)
                .descriptor_count(count)
                .build(),
        );
        self
    }

    pub fn build(self, graphics_device: &Device, max_sets: u32) -> Option<UniqueDescriptorPool> {
        unsafe {
            graphics_device.create_descriptor_pool(
                &DescriptorPoolCreateInfo::builder()
                    .max_sets(max_sets)
                    .pool_sizes(&self.pools)
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create descriptor pool: {}", e))
        .map(|dpool| UniqueDescriptorPool {
            dpool,
            device: graphics_device as *const _,
        })
        .ok()
    }
}

impl std::ops::Drop for UniqueDescriptorPool {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_descriptor_pool(self.dpool, None);
        }
    }
}

pub struct UniqueSwapchain {
    pub swapchain: SwapchainKHR,
    loader: *const Swapchain,
}

impl std::ops::Drop for UniqueSwapchain {
    fn drop(&mut self) {
        unsafe { (*self.loader).destroy_swapchain(self.swapchain, None) }
    }
}

impl UniqueSwapchain {
    pub fn new(
        swapchain_loader: &Swapchain,
        surface_caps: &SurfaceCapabilitiesKHR,
        vk_surface: SurfaceKHR,
        surface_format: SurfaceFormatKHR,
        presentation_mode: PresentModeKHR,
    ) -> Option<(UniqueSwapchain, u32)> {
        let image_count = (surface_caps.min_image_count + 1)
            .clamp(surface_caps.min_image_count, surface_caps.max_image_count);

        unsafe {
            swapchain_loader.create_swapchain(
                &SwapchainCreateInfoKHR::builder()
                    .min_image_count(image_count)
                    .surface(vk_surface)
                    .image_format(surface_format.format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(surface_caps.current_extent)
                    .image_array_layers(1)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .pre_transform(surface_caps.current_transform)
                    .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(presentation_mode)
                    .clipped(true)
                    .build(),
                None,
            )
        }
        .map_err(|e| {
            error!("Failed to create swapchain {}", e);
        })
        .map(move |swap_chain| {
            (
                UniqueSwapchain {
                    swapchain: swap_chain,
                    loader: swapchain_loader as *const _,
                },
                image_count,
            )
        })
        .ok()
    }
}

pub struct UniqueBuffer {
    pub buffer: Buffer,
    pub memory: DeviceMemory,
    device: *const Device,
}

impl std::ops::Drop for UniqueBuffer {
    fn drop(&mut self) {
        unsafe {
            (*self.device).free_memory(self.memory, None);
            (*self.device).destroy_buffer(self.buffer, None);
        }
    }
}

impl UniqueBuffer {
    pub fn new(
        renderer: &VulkanRenderer,
        usage: BufferUsageFlags,
        memory_type: MemoryPropertyFlags,
        bytes: DeviceSize,
    ) -> Option<UniqueBuffer> {
        let graphics_device = renderer.graphics_device();
        let memory_props = renderer.device_memory();

        let buffer = unsafe {
            graphics_device.create_buffer(
                &BufferCreateInfo::builder()
                    .size(bytes)
                    .usage(usage)
                    .sharing_mode(SharingMode::EXCLUSIVE)
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create buffer: {}", e))
        .ok()?;

        let memory_requirements = unsafe { graphics_device.get_buffer_memory_requirements(buffer) };

        let memory = unsafe {
            graphics_device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(choose_memory_type(
                        memory_props,
                        &memory_requirements,
                        memory_type,
                    ))
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to allocate meory for buffer: {}", e))
        .ok()?;

        unsafe { graphics_device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| error!("Failed to bind device memory to buffer: {}", e))
            .ok()?;

        Some(UniqueBuffer {
            buffer,
            memory,
            device: graphics_device as *const _,
        })
    }

    pub fn gpu_only_buffer<T: Sized>(
        renderer: &VulkanRenderer,
        usage: BufferUsageFlags,
        memory_type: MemoryPropertyFlags,
        data: &[&[T]],
        alignment: Option<u64>,
    ) -> Option<UniqueBuffer> {
        let items_count = data
            .iter()
            .fold(0u64, |count, data_chunk| count + data_chunk.len() as u64);

        let bytes_size = alignment
            .map(|align| VulkanRenderer::aligned_size_of_type::<T>(align) * items_count)
            .unwrap_or_else(|| items_count * size_of::<T>() as u64);

        let gpu_buffer = Self::new(
            renderer,
            usage | BufferUsageFlags::TRANSFER_DST,
            memory_type,
            bytes_size,
        )?;
        let staging_buffer = Self::new(
            renderer,
            usage | BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE,
            bytes_size,
        )?;

        //
        // CPU -> GPU staging buffer
        ScopedBufferMapping::create(renderer, &staging_buffer, WHOLE_SIZE, 0).map(|mapping| {
            let _ = data.iter().fold(0, |items, src_buf| {
                unsafe {
                    copy_nonoverlapping(
                        src_buf.as_ptr(),
                        (mapping.memptr() as *mut T).offset(items as isize),
                        src_buf.len(),
                    );
                }
                items + src_buf.len()
            });
        })?;

        //
        // GPU staging -> GPU only
        unsafe {
            renderer.graphics_device().cmd_copy_buffer(
                renderer.res_loader().cmd_buf,
                staging_buffer.buffer,
                gpu_buffer.buffer,
                &[BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(bytes_size)
                    .build()],
            );
        }

        renderer.res_loader.add_staging_buffer(staging_buffer);
        Some(gpu_buffer)
    }
}

pub struct Cpu2GpuBuffer<T: Sized> {
    pub buffer: UniqueBuffer,
    pub align: DeviceSize,
    pub bytes_one_frame: DeviceSize,
    _phantom: std::marker::PhantomData<*const T>,
}

impl<T: Sized> Cpu2GpuBuffer<T> {
    pub fn create(
        renderer: &VulkanRenderer,
        usage: BufferUsageFlags,
        align: DeviceSize,
        items: DeviceSize,
        max_frames: DeviceSize,
    ) -> Option<Self> {
        let align = VulkanRenderer::aligned_size_of_type::<T>(align);
        Some(Self {
            buffer: UniqueBuffer::new(
                renderer,
                usage,
                MemoryPropertyFlags::HOST_VISIBLE,
                align * items * max_frames,
            )?,
            align,
            bytes_one_frame: align * items,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn map_for_frame(
        &self,
        renderer: &VulkanRenderer,
        frame_id: DeviceSize,
    ) -> Option<ScopedBufferMapping> {
        ScopedBufferMapping::create(
            renderer,
            &self.buffer,
            self.bytes_one_frame,
            self.bytes_one_frame * frame_id,
        )
    }

    pub fn offset_for_frame(&self, frame_id: DeviceSize) -> DeviceSize {
        self.bytes_one_frame * frame_id
    }
}

pub struct ScopedBufferMapping<'a> {
    buffer: &'a UniqueBuffer,
    memptr: *mut c_void,
    size: DeviceSize,
    offset: DeviceSize,
}

impl<'a> ScopedBufferMapping<'a> {
    pub fn create(
        renderer: &VulkanRenderer,
        buffer: &'a UniqueBuffer,
        map_size: DeviceSize,
        map_offset: DeviceSize,
    ) -> Option<ScopedBufferMapping<'a>> {
        let graphics_device = renderer.graphics_device();
        let mapped_memory = unsafe {
            graphics_device.map_memory(buffer.memory, map_offset, map_size, MemoryMapFlags::empty())
        }
        .map_err(|e| error!("Failed to map buffer memory: {}", e))
        .ok()?;

        Some(ScopedBufferMapping {
            buffer,
            memptr: mapped_memory,
            size: map_size,
            offset: map_offset,
        })
    }

    pub fn memptr(&self) -> *mut c_void {
        self.memptr
    }
}

impl<'a> std::ops::Drop for ScopedBufferMapping<'a> {
    fn drop(&mut self) {
        unsafe {
            let mapped_mem_ranges = [MappedMemoryRange::builder()
                .memory(self.buffer.memory)
                .offset(self.offset)
                .size(self.size)
                .build()];
            let _ = (*self.buffer.device)
                .flush_mapped_memory_ranges(&mapped_mem_ranges)
                .map_err(|e| error!("Error flushing mapped memory: {}", e));
            (*self.buffer.device).unmap_memory(self.buffer.memory);
        }
    }
}

impl<'a> std::ops::Deref for ScopedBufferMapping<'a> {
    type Target = [u8];

    fn deref(&self) -> &'a Self::Target {
        unsafe { std::slice::from_raw_parts(self.memptr as *const u8, self.size as usize) }
    }
}

impl<'a> std::ops::DerefMut for ScopedBufferMapping<'a> {
    fn deref_mut(&mut self) -> &'a mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.memptr as *mut u8, self.size as usize) }
    }
}

pub struct UniqueSampler {
    pub sampler: ash::vk::Sampler,
    device: *const Device,
}

impl UniqueSampler {
    pub fn new(graphics_device: &Device, create_info: &SamplerCreateInfo) -> Option<UniqueSampler> {
        let sampler = unsafe { graphics_device.create_sampler(create_info, None) }
            .map_err(|e| error!("Failed to create sampler: {}", e))
            .ok()?;

        Some(UniqueSampler {
            sampler,
            device: graphics_device as *const _,
        })
    }
}

impl std::ops::Drop for UniqueSampler {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_sampler(self.sampler, None);
        }
    }
}

pub struct GraphicsPipelineLayoutBuilder {
    pub layout_bindings: std::collections::hash_map::HashMap<u32, Vec<DescriptorSetLayoutBinding>>,
    current_set: u32,
    pub push_constants: Vec<PushConstantRange>,
}

impl GraphicsPipelineLayoutBuilder {
    pub fn new() -> Self {
        Self {
            layout_bindings: HashMap::new(),
            current_set: 0u32,
            push_constants: Vec::new(),
        }
    }

    pub fn set(mut self, set_id: u32, bindings: &[DescriptorSetLayoutBinding]) -> Self {
        self.layout_bindings
            .entry(set_id)
            .and_modify(|set_entry| set_entry.extend_from_slice(bindings))
            .or_insert(Vec::from(bindings));

        self
    }

    pub fn add_push_constant(mut self, pushc: PushConstantRange) -> Self {
        self.push_constants.push(pushc);
        self
    }

    pub fn build(
        self,
        graphics_device: &Device,
    ) -> Option<(PipelineLayout, Vec<DescriptorSetLayout>)> {
        let mut layout_descriptions = self.layout_bindings.into_iter().collect::<Vec<_>>();
        layout_descriptions.sort_unstable_by_key(|(set_id, _)| *set_id);

        let descriptor_set_layouts = layout_descriptions
            .iter()
            .filter_map(|(_, set_bindings)| unsafe {
                graphics_device
                    .create_descriptor_set_layout(
                        &DescriptorSetLayoutCreateInfo::builder()
                            .flags(DescriptorSetLayoutCreateFlags::empty())
                            .bindings(&set_bindings)
                            .build(),
                        None,
                    )
                    .map_err(|e| error!("Failed to create descriptor set layout: {}", e))
                    .ok()
            })
            .collect::<Vec<_>>();

        if descriptor_set_layouts.len() != layout_descriptions.len() {
            error!("Failed to create all required descriptor set layouts");
            return None;
        }

        unsafe {
            graphics_device.create_pipeline_layout(
                &PipelineLayoutCreateInfo::builder()
                    .push_constant_ranges(&self.push_constants)
                    .set_layouts(&descriptor_set_layouts)
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create pipeline layout; {}", e))
        .map(|pipeline_layout| (pipeline_layout, descriptor_set_layouts))
        .ok()
    }
}

pub struct UniqueShaderModule {
    pub handle: ShaderModule,
    device: *const Device,
}

impl std::ops::Drop for UniqueShaderModule {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_shader_module(self.handle, None);
        }
    }
}

impl UniqueShaderModule {
    pub fn from_bytecode(graphics_device: &Device, bytecode: &[u8]) -> Option<UniqueShaderModule> {
        let bytecode = unsafe {
            std::slice::from_raw_parts(
                bytecode.as_ptr() as *const _ as *const u32,
                bytecode.len() / size_of::<u32>(),
            )
        };

        unsafe {
            graphics_device.create_shader_module(
                &ShaderModuleCreateInfo::builder().code(bytecode).build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create shader module {}", e))
        .map(|module| UniqueShaderModule {
            handle: module,
            device: graphics_device as *const _,
        })
        .ok()
    }

    pub fn from_file<P: AsRef<std::path::Path> + ?Sized>(
        graphics_device: &Device,
        path: &P,
    ) -> Option<UniqueShaderModule> {
        let bytecode_file = std::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| {
                error!(
                    "Failed to open shader file {}, error:\n{}",
                    path.as_ref().to_str().unwrap(),
                    e
                )
            })
            .ok()?;

        let metadata = bytecode_file.metadata().ok()?;
        let mapped_file = unsafe {
            mmap_rs::MmapOptions::new(metadata.len() as usize)
                .with_file(bytecode_file, 0)
                .map()
        }
        .map_err(|e| {
            error!(
                "Failed to memory map shader bytecode file {}, error: {}",
                path.as_ref().to_str().unwrap(),
                e
            )
        })
        .ok()?;

        UniqueShaderModule::from_bytecode(graphics_device, &mapped_file)
    }
}

#[derive(Clone)]
pub enum ShaderModuleSource<'a> {
    Bytes(&'a [u8]),
    File(&'a std::path::Path),
}

#[derive(Clone)]
pub struct ShaderModuleDescription<'a> {
    pub stage: ShaderStageFlags,
    pub source: ShaderModuleSource<'a>,
    pub entry_point: &'a str,
}

pub struct GraphicsPipelineBuilder<'a> {
    shader_stages: Vec<ShaderModuleDescription<'a>>,
    vertex_input_attrib_desc: Vec<VertexInputAttributeDescription>,
    vertex_input_attrib_bindings: Vec<VertexInputBindingDescription>,
    input_assembly_state: PipelineInputAssemblyStateCreateInfo,
    viewports: Vec<Viewport>,
    scissors: Vec<Rect2D>,
    raster_state: PipelineRasterizationStateCreateInfo,
    multisample_state: PipelineMultisampleStateCreateInfo,
    depth_stencil_state: PipelineDepthStencilStateCreateInfo,
    colorblend_state: Vec<PipelineColorBlendAttachmentState>,
    dynamic_state: Vec<DynamicState>,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new() -> Self {
        Self {
            shader_stages: Vec::new(),
            vertex_input_attrib_desc: Vec::new(),
            vertex_input_attrib_bindings: Vec::new(),
            input_assembly_state: PipelineInputAssemblyStateCreateInfo::builder()
                .topology(PrimitiveTopology::TRIANGLE_LIST)
                .build(),
            viewports: Vec::new(),
            scissors: Vec::new(),
            raster_state: PipelineRasterizationStateCreateInfo::builder()
                .cull_mode(CullModeFlags::BACK)
                .front_face(FrontFace::COUNTER_CLOCKWISE)
                .polygon_mode(PolygonMode::FILL)
                .line_width(1f32)
                .build(),
            multisample_state: PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(SampleCountFlags::TYPE_1)
                .build(),
            depth_stencil_state: PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .stencil_test_enable(false)
                .depth_compare_op(CompareOp::LESS)
                .depth_write_enable(true)
                .min_depth_bounds(1f32)
                .max_depth_bounds(0f32)
                .build(),
            colorblend_state: vec![PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(BlendOp::ADD)
                .src_alpha_blend_factor(BlendFactor::ONE)
                .dst_alpha_blend_factor(BlendFactor::ZERO)
                .alpha_blend_op(BlendOp::ADD)
                .color_write_mask(ColorComponentFlags::RGBA)
                .build()],
            dynamic_state: Vec::new(),
        }
    }

    pub fn shader_stages(mut self, shader_modules: &'a [ShaderModuleDescription]) -> Self {
        self.shader_stages.extend_from_slice(shader_modules);
        self
    }

    pub fn add_vertex_input_attribute_descriptions(
        mut self,
        attributes: &[VertexInputAttributeDescription],
    ) -> Self {
        self.vertex_input_attrib_desc.extend_from_slice(attributes);
        self
    }

    pub fn add_vertex_input_attribute_binding(
        mut self,
        binding: VertexInputBindingDescription,
    ) -> Self {
        self.vertex_input_attrib_bindings.push(binding);
        self
    }

    pub fn set_input_assembly_state(
        mut self,
        topology: PrimitiveTopology,
        enable_restart: bool,
    ) -> Self {
        self.input_assembly_state.topology = topology;
        self.input_assembly_state.primitive_restart_enable = enable_restart as Bool32;

        self
    }

    pub fn set_rasterization_state(mut self, state: PipelineRasterizationStateCreateInfo) -> Self {
        self.raster_state = state;
        self
    }

    pub fn set_depth_compare_op(mut self, cmp_op: CompareOp) -> Self {
        self.depth_stencil_state.depth_compare_op = cmp_op;
        self
    }

    pub fn build(
        self,
        graphics_device: &Device,
        pipeline_cache: PipelineCache,
        pipeline_layout_with_descriptor_layouts: (PipelineLayout, Vec<DescriptorSetLayout>),
        renderpass: RenderPass,
        subpass: u32,
    ) -> Option<UniqueGraphicsPipeline> {
        let (pipeline_layout, descriptor_set_layouts) = pipeline_layout_with_descriptor_layouts;

        let built_shader_modules = self
            .shader_stages
            .iter()
            .filter_map(|smi| {
                let bytecode = match &smi.source {
                    ShaderModuleSource::File(path) => {
                        UniqueShaderModule::from_file(graphics_device, *path)
                    }
                    ShaderModuleSource::Bytes(bytecode) => {
                        UniqueShaderModule::from_bytecode(graphics_device, bytecode)
                    }
                }?;

                let entry_point =
                    CString::new(smi.entry_point.clone()).expect("Failed conversion to CString");

                Some((bytecode, entry_point))
            })
            .collect::<Vec<_>>();

        if built_shader_modules.len() != self.shader_stages.len() {
            info!("Failed to build all required shader modules");
            return None;
        }

        let shader_stages_create_info = self
            .shader_stages
            .iter()
            .zip(built_shader_modules.iter())
            .map(|(mi, (module, entry_pt))| {
                PipelineShaderStageCreateInfo::builder()
                    .name(entry_pt.as_c_str())
                    .stage(mi.stage)
                    .module(module.handle)
                    .build()
            })
            .collect::<Vec<_>>();

        let mut vsb = PipelineViewportStateCreateInfo::builder();
        if self.viewports.is_empty() {
            vsb = vsb.viewport_count(1);
        } else {
            vsb = vsb.viewports(&self.viewports);
        }

        if self.scissors.is_empty() {
            vsb = vsb.scissor_count(1);
        } else {
            vsb = vsb.scissors(&self.scissors);
        }

        let pipelines_create_info = [GraphicsPipelineCreateInfo::builder()
            .input_assembly_state(&self.input_assembly_state)
            .vertex_input_state(
                &PipelineVertexInputStateCreateInfo::builder()
                    .vertex_attribute_descriptions(&self.vertex_input_attrib_desc)
                    .vertex_binding_descriptions(&self.vertex_input_attrib_bindings)
                    .build(),
            )
            .stages(&shader_stages_create_info)
            .depth_stencil_state(&self.depth_stencil_state)
            .multisample_state(&self.multisample_state)
            .rasterization_state(&self.raster_state)
            .dynamic_state(
                &PipelineDynamicStateCreateInfo::builder().dynamic_states(&self.dynamic_state),
            )
            .layout(pipeline_layout)
            .render_pass(renderpass)
            .subpass(subpass)
            .viewport_state(&vsb.build())
            .color_blend_state(
                &PipelineColorBlendStateCreateInfo::builder()
                    .attachments(&self.colorblend_state)
                    .blend_constants([1f32; 4])
                    .build(),
            )
            .build()];

        unsafe {
            graphics_device.create_graphics_pipelines(pipeline_cache, &pipelines_create_info, None)
        }
        .map_err(|e| error!("Failed to create graphics pipeline: {:?}", e))
        .map(|pipelines| UniqueGraphicsPipeline {
            pipeline: pipelines[0],
            layout: pipeline_layout,
            descriptor_layouts: descriptor_set_layouts,
            device: graphics_device as *const _,
        })
        .ok()
    }

    pub fn set_multisample_state(
        mut self,
        multisample_state: PipelineMultisampleStateCreateInfo,
    ) -> Self {
        self.multisample_state = multisample_state;
        self
    }

    pub fn set_depth_stencil_state(
        mut self,
        depth_stencil_state: PipelineDepthStencilStateCreateInfo,
    ) -> Self {
        self.depth_stencil_state = depth_stencil_state;
        self
    }

    pub fn set_depth_test(mut self, enabled: bool) -> Self {
        self.depth_stencil_state.depth_test_enable = enabled as u32;
        self
    }

    pub fn set_colorblend_attachment(
        mut self,
        idx: usize,
        colorblend_state: PipelineColorBlendAttachmentState,
    ) -> Self {
        self.colorblend_state[idx] = colorblend_state;
        self
    }

    pub fn add_colorblend_attachment(
        mut self,
        colorblend_state: PipelineColorBlendAttachmentState,
    ) -> Self {
        self.colorblend_state.push(colorblend_state);
        self
    }

    pub fn add_dynamic_state(mut self, dynamic_state: DynamicState) -> Self {
        self.dynamic_state.push(dynamic_state);
        self
    }

    pub fn dynamic_states(mut self, dyn_states: &[DynamicState]) -> Self {
        self.dynamic_state.extend_from_slice(dyn_states);
        self
    }
}

pub struct UniqueGraphicsPipeline {
    pub pipeline: Pipeline,
    pub layout: PipelineLayout,
    descriptor_layouts: Vec<DescriptorSetLayout>,
    device: *const Device,
}

impl UniqueGraphicsPipeline {
    pub fn descriptor_layouts(&self) -> &[DescriptorSetLayout] {
        &self.descriptor_layouts
    }
}

impl std::ops::Drop for UniqueGraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_pipeline_layout(self.layout, None);
            (*self.device).destroy_pipeline(self.pipeline, None);
        }
        self.descriptor_layouts
            .iter()
            .for_each(|&desc_set_layout| unsafe {
                (*self.device).destroy_descriptor_set_layout(desc_set_layout, None);
            });
    }
}

pub struct UniquePipelineCache {
    pub cache: PipelineCache,
    device: *const Device,
}

impl UniquePipelineCache {
    pub fn new(graphics_device: &Device) -> Option<UniquePipelineCache> {
        unsafe {
            graphics_device.create_pipeline_cache(&PipelineCacheCreateInfo::builder().build(), None)
        }
        .map_err(|e| error!("Failed to create pipeline cache: {}", e))
        .map(|cache| UniquePipelineCache {
            cache,
            device: graphics_device as *const _,
        })
        .ok()
    }
}

impl std::ops::Drop for UniquePipelineCache {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_pipeline_cache(self.cache, None);
        }
    }
}

pub struct UniqueRenderpass {
    pub handle: RenderPass,
    device: *const Device,
}

impl UniqueRenderpass {
    pub fn new(
        graphics_device: &Device,
        surface_fmt: Format,
        depth_stencil_fmt: Format,
    ) -> Option<UniqueRenderpass> {
        let attachment_descriptions = [
            AttachmentDescription::builder()
                .format(surface_fmt)
                .samples(SampleCountFlags::TYPE_1)
                .load_op(AttachmentLoadOp::CLEAR)
                .store_op(AttachmentStoreOp::STORE)
                .stencil_load_op(AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(AttachmentStoreOp::DONT_CARE)
                .initial_layout(ImageLayout::UNDEFINED)
                .final_layout(ImageLayout::PRESENT_SRC_KHR)
                .build(),
            AttachmentDescription::builder()
                .format(depth_stencil_fmt)
                .samples(SampleCountFlags::TYPE_1)
                .load_op(AttachmentLoadOp::CLEAR)
                .store_op(AttachmentStoreOp::STORE)
                .stencil_store_op(AttachmentStoreOp::STORE)
                .stencil_load_op(AttachmentLoadOp::LOAD)
                .initial_layout(ImageLayout::UNDEFINED)
                .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];

        let attachment_references = [
            AttachmentReference::builder()
                .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .attachment(0)
                .build(),
            AttachmentReference::builder()
                .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .attachment(1)
                .build(),
        ];

        let dependencies = [SubpassDependency::builder()
            .dst_subpass(0)
            .src_subpass(SUBPASS_EXTERNAL)
            .src_access_mask(AccessFlags::NONE)
            .dst_access_mask(
                AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .src_stage_mask(
                PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .build()];

        let subpass_descriptions = [SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_references[0..1])
            .depth_stencil_attachment(&attachment_references[1])
            .build()];

        unsafe {
            graphics_device.create_render_pass(
                &RenderPassCreateInfo::builder()
                    .subpasses(&subpass_descriptions)
                    .attachments(&attachment_descriptions)
                    .dependencies(&dependencies)
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to create renderpass: {}", e))
        .ok()
        .map(|renderpass| UniqueRenderpass {
            handle: renderpass,
            device: graphics_device as *const _,
        })
    }
}

impl std::ops::Drop for UniqueRenderpass {
    fn drop(&mut self) {
        unsafe {
            (*self.device).destroy_render_pass(self.handle, None);
        }
    }
}

pub struct FrameRenderData {
    pub swapchain_image: Image,
    pub command_buffer: CommandBuffer,
    pub swapchain_image_view: UniqueImageView,
    pub framebuffer: UniqueFramebuffer,
    pub depth_stencil_tex_view: UniqueImageView,
    pub depth_stencil_tex: UniqueImage,
    pub fence: UniqueFence,
    pub sem_img_available: UniqueSemaphore,
    pub sem_img_rending_done: UniqueSemaphore,
}

impl FrameRenderData {
    fn new(
        memory_props: &PhysicalDeviceMemoryProperties,
        swapchain_image: Image,
        graphics_device: &Device,
        cmd_pool: CommandPool,
        depth_stencil_fmt: Format,
        swapchain_fmt: Format,
        width: u32,
        height: u32,
        renderpass: RenderPass,
    ) -> Option<FrameRenderData> {
        let cmd_buf = unsafe {
            graphics_device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .command_pool(cmd_pool)
                    .build(),
            )
        }
        .map_err(|e| error!("Failed to create command buffer {}", e))
        .ok()?;

        let swapchain_imageview = UniqueImageView::new(
            graphics_device,
            &ImageViewCreateInfo::builder()
                .format(swapchain_fmt)
                .image(swapchain_image)
                .view_type(ImageViewType::TYPE_2D)
                .format(swapchain_fmt)
                .components(
                    ComponentMapping::builder()
                        .r(ComponentSwizzle::R)
                        .g(ComponentSwizzle::G)
                        .b(ComponentSwizzle::B)
                        .a(ComponentSwizzle::A)
                        .build(),
                )
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .level_count(1)
                        .layer_count(1)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .build(),
                )
                .build(),
        )?;

        let (ds_image, ds_image_view) = FrameRenderData::create_depth_stencil_buffer(
            graphics_device,
            memory_props,
            depth_stencil_fmt,
            width,
            height,
        )?;

        let fb_attachments = [swapchain_imageview.view, ds_image_view.view];

        let framebuffer = UniqueFramebuffer::new(
            &graphics_device,
            &FramebufferCreateInfo::builder()
                .attachments(&fb_attachments)
                .render_pass(renderpass)
                .width(width)
                .height(height)
                .layers(1)
                .build(),
        )?;

        let fence = UniqueFence::new(&graphics_device, true)?;

        let sem_img_avail = UniqueSemaphore::new(graphics_device)?;
        let sem_rendering_done = UniqueSemaphore::new(graphics_device)?;

        Some(FrameRenderData {
            swapchain_image,
            command_buffer: cmd_buf[0],
            swapchain_image_view: swapchain_imageview,
            depth_stencil_tex: ds_image,
            depth_stencil_tex_view: ds_image_view,
            framebuffer,
            fence,
            sem_img_available: sem_img_avail,
            sem_img_rending_done: sem_rendering_done,
        })
    }

    fn create_depth_stencil_buffer(
        graphics_device: &Device,
        memory_props: &PhysicalDeviceMemoryProperties,
        format: Format,
        width: u32,
        height: u32,
    ) -> Option<(UniqueImage, UniqueImageView)> {
        let image_create_info = ImageCreateInfo::builder()
            .initial_layout(ImageLayout::UNDEFINED)
            .samples(SampleCountFlags::TYPE_1)
            .format(format)
            .usage(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .tiling(ImageTiling::OPTIMAL)
            .image_type(ImageType::TYPE_2D)
            .extent(Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .build();

        let image = unsafe { graphics_device.create_image(&image_create_info, None) }
            .map_err(|e| error!("Can't create depth stencil image: {}", e))
            .ok()?;

        let memory_requirements = unsafe { graphics_device.get_image_memory_requirements(image) };

        let image_memory = unsafe {
            graphics_device.allocate_memory(
                &MemoryAllocateInfo::builder()
                    .memory_type_index(choose_memory_type(
                        memory_props,
                        &memory_requirements,
                        MemoryPropertyFlags::DEVICE_LOCAL,
                    ))
                    .allocation_size(memory_requirements.size)
                    .build(),
                None,
            )
        }
        .map_err(|e| error!("Failed to allocate memory for image: {}", e))
        .ok()?;

        unsafe { graphics_device.bind_image_memory(image, image_memory, 0) }
            .map_err(|e| error!("Failed to bind memory for image: {}", e))
            .ok()?;

        let image_view = UniqueImageView::new(
            graphics_device,
            &ImageViewCreateInfo::builder()
                .image(image)
                .view_type(ImageViewType::TYPE_2D)
                .format(format)
                .components(
                    ComponentMapping::builder()
                        .r(ComponentSwizzle::R)
                        .g(ComponentSwizzle::G)
                        .b(ComponentSwizzle::B)
                        .a(ComponentSwizzle::A)
                        .build(),
                )
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .aspect_mask(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL)
                        .base_mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                )
                .build(),
        )?;

        Some((
            UniqueImage {
                image,
                memory: image_memory,
                device: graphics_device as *const _,
                info: image_create_info.into(),
            },
            image_view,
        ))
    }
}

pub struct ResourceLoader {
    pub cmd_buf: CommandBuffer,
    fence: UniqueFence,
    staging_buffers: RefCell<Vec<UniqueBuffer>>,
}

impl ResourceLoader {
    pub fn new(graphics_device: &Device, cmd_pool: CommandPool) -> Option<ResourceLoader> {
        let cmd_buffers = unsafe {
            graphics_device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .command_pool(cmd_pool)
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .build(),
            )
        }
        .map_err(|e| error!("Failed to create resource loading command buffer: {}", e))
        .ok()?;

        Some(ResourceLoader {
            cmd_buf: cmd_buffers[0],
            fence: UniqueFence::new(graphics_device, false)?,
            staging_buffers: RefCell::new(Vec::new()),
        })
    }

    pub fn add_staging_buffer(&self, staging_buf: UniqueBuffer) {
        self.staging_buffers.borrow_mut().push(staging_buf);
    }
}

pub struct RendererWorkPackage {
    pub cmd_buf: CommandBuffer,
    pub wait_fence: UniqueFence,
}

impl RendererWorkPackage {
    pub fn wait_for_completion(&self, renderer: &VulkanRenderer) {
        unsafe {
            let fences = [self.wait_fence.fence];
            let _ = renderer
                .graphics_device()
                .wait_for_fences(&fences, true, !0u64);
        }
    }

    pub fn submit(self, renderer: &VulkanRenderer) {
        renderer.push_work_package(self);
    }
}

pub struct VulkanRenderer {
    work_packages: RefCell<Vec<RendererWorkPackage>>,
    pipeline_cache: UniquePipelineCache,
    res_loader: ResourceLoader,
    renderpass: UniqueRenderpass,
    current_frame_id: std::cell::Cell<u32>,
    frame_render_data: Vec<FrameRenderData>,
    framebuffer_extents: Extent2D,
    surface_format: Format,
    depth_stencil_format: Format,
    cmd_pool: UniqueCommandPool,
    descriptor_pool: UniqueDescriptorPool,
    queue: Queue,
    queue_family_index: u32,
    max_inflight_frames: u32,
    swapchain: UniqueSwapchain,
    swapchain_loader: std::pin::Pin<std::boxed::Box<Swapchain>>,
    graphics_device: std::pin::Pin<std::boxed::Box<Device>>,
    device_memory: PhysicalDeviceMemoryProperties,
    device_properties: PhysicalDeviceProperties,
    device_features: PhysicalDeviceFeatures,
    debug_utils_msg: DebugUtilsMessengerEXT,
    debug_utils: DebugUtils,
    surface_loader: Surface,
    vk_surface: SurfaceKHR,
    phys_device: PhysicalDevice,
    vk_instance: Instance,
    vk_entry: Entry,
}

impl VulkanRenderer {
    #[cfg(target_os = "windows")]
    fn create_vulkan_surface_winit(
        window: &winit::window::Window,
        vk_instance: &ash::Instance,
        vk_entry: &ash::Entry,
    ) -> VkResult<vk::SurfaceKHR> {
        let win32_module_handle = unsafe {
            std::mem::transmute::<windows_sys::Win32::Foundation::HINSTANCE, ash::vk::HINSTANCE>(
                windows_sys::Win32::System::LibraryLoader::GetModuleHandleA(std::ptr::null()),
            )
        };

        let win32_surface = ash::extensions::khr::Win32Surface::new(vk_entry, vk_instance);
        unsafe {
            use winit::platform::windows::WindowExtWindows;

            win32_surface.create_win32_surface(
                &vk::Win32SurfaceCreateInfoKHR::builder()
                    .hwnd(std::mem::transmute::<
                        winit::platform::windows::HWND,
                        ash::vk::HWND,
                    >(window.hwnd()))
                    .hinstance(win32_module_handle)
                    .build(),
                None,
            )
        }
    }

    #[cfg(target_family = "unix")]
    fn create_vulkan_surface_winit(
        win: &winit::window::Window,
        vk_instance: &ash::Instance,
        vk_entry: &ash::Entry,
    ) -> VkResult<vk::SurfaceKHR> {
        use std::mem::transmute;
        use winit::platform::unix::WindowExtUnix;

        let native_window = win.xlib_window().expect("Failed to query native window id");
        let native_display = win.xlib_display().expect("Failed to query native display");

        let xlib_surface = ash::extensions::khr::XlibSurface::new(vk_entry, vk_instance);
        unsafe {
            xlib_surface.create_xlib_surface(
                &vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(native_display as *mut ash::vk::Display)
                    .window(transmute::<u64, ash::vk::Window>(native_window))
                    .build(),
                None,
            )
        }
    }

    pub fn push_staging_buffer(&self, staging_buffer: UniqueBuffer) {
        self.res_loader.add_staging_buffer(staging_buffer);
    }

    pub fn create_work_package(&self) -> Option<RendererWorkPackage> {
        let allocated_cmd_buffers = unsafe {
            self.graphics_device.allocate_command_buffers(
                &CommandBufferAllocateInfo::builder()
                    .level(CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1)
                    .command_pool(self.cmd_pool.cmd_pool)
                    .build(),
            )
        }
        .map_err(|e| {
            error!(
                "Failed to allocate command buffer for work package, error: {}",
                e
            )
        })
        .ok()?;

        let cmd_buff = allocated_cmd_buffers[0];

        unsafe {
            self.graphics_device
                .begin_command_buffer(
                    cmd_buff,
                    &CommandBufferBeginInfo::builder()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )
                .expect("Failed to begin command buffer");
        }

        let wait_fence = UniqueFence::new(&self.graphics_device, false)?;

        Some(RendererWorkPackage {
            cmd_buf: allocated_cmd_buffers[0],
            wait_fence,
        })
    }

    pub fn push_work_package(&self, work_package: RendererWorkPackage) {
        unsafe {
            self.graphics_device
                .end_command_buffer(work_package.cmd_buf)
        }
        .expect("Failed to end command buffer");

        let command_buffers = [work_package.cmd_buf];
        let this_submit = [SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build()];
        unsafe {
            self.graphics_device.queue_submit(
                self.queue,
                &this_submit,
                work_package.wait_fence.fence,
            )
        }
        .expect("Failed to submit work package");

        self.work_packages.borrow_mut().push(work_package);
    }

    pub fn wait_all_work_packages(&self) {
        info!(
            "Waiting for {} GPU work packages",
            self.work_packages.borrow().len()
        );

        let wait_fences = self
            .work_packages
            .borrow()
            .iter()
            .map(|work_pkg| work_pkg.wait_fence.fence)
            .collect::<smallvec::SmallVec<[Fence; 16]>>();

        unsafe {
            self.graphics_device
                .wait_for_fences(&wait_fences, true, !0u64)
                .expect("Failed to wait for fences...");
        }

        self.work_packages.borrow_mut().clear();
    }

    pub fn create(window: &winit::window::Window) -> Option<VulkanRenderer> {
        let vk_entry = Entry::linked();
        let validation_layer_name =
            CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();

        vk_entry
            .enumerate_instance_layer_properties()
            .map(|layer_properties| {
                layer_properties.iter().any(|layer| {
                    let layer_name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    layer_name == validation_layer_name
                })
            })
            .ok()?;

        #[cfg(target_os = "windows")]
        let required_extensions = [
            vk::KhrSurfaceFn::name(),
            vk::ExtDebugUtilsFn::name(),
            vk::KhrWin32SurfaceFn::name(),
        ];

        #[cfg(target_family = "unix")]
        let required_extensions = [
            vk::KhrSurfaceFn::name(),
            vk::ExtDebugUtilsFn::name(),
            vk::KhrXlibSurfaceFn::name(),
        ];

        let has_all_required_exts = vk_entry
            .enumerate_instance_extension_properties(None)
            .ok()
            .map(|instance_extensions| {
                info!("Instance extensions present:");

                let inst_exts_names = instance_extensions
                    .iter()
                    .map(|inst_ext| {
                        let ext_name = unsafe { CStr::from_ptr(inst_ext.extension_name.as_ptr()) };
                        info!(
                            "{}",
                            ext_name.to_str().unwrap_or("cannot display extension name")
                        );
                        ext_name
                    })
                    .collect::<Vec<_>>();

                required_extensions
                    .iter()
                    .all(|&req_ext| inst_exts_names.iter().any(|&inst_ext| inst_ext == req_ext))
            })
            .unwrap_or(false);

        if !has_all_required_exts {
            error!("Vulkan instance does not support all required extensions");
            return None;
        }

        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 3, 0))
            .application_name(&std::ffi::CString::new("vulkan-experiments-rust").unwrap())
            .application_version(0)
            .engine_name(&std::ffi::CString::new("Vulkan-B5-RS").unwrap())
            .build();

        let enabled_layers_names = [validation_layer_name.as_ptr()];
        let enabled_extension_names = required_extensions
            .iter()
            .map(|req_ext| req_ext.as_ptr())
            .collect::<Vec<_>>();

        let mut debug_utils_messenger_create_info = DebugUtilsMessengerCreateInfoEXT::builder()
            .message_type(
                DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .message_severity(
                DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .pfn_user_callback(Some(debug_message_callback))
            .build();

        let vk_instance = unsafe {
            vk_entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&enabled_extension_names)
                    .enabled_layer_names(&enabled_layers_names)
                    .push_next(&mut debug_utils_messenger_create_info)
                    .build(),
                None,
            )
        }
        .map_err(|e| {
            error!("Failed to create Vulkan instance {}", e);
        })
        .ok()?;

        let debug_utils = ash::extensions::ext::DebugUtils::new(&vk_entry, &vk_instance);
        let debug_utils_msg = unsafe {
            debug_utils.create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
        }
        .map_err(|e| {
            error!("Failed to create debug utils messenger:: {}", e);
        })
        .ok()?;

        let surface_loader = ash::extensions::khr::Surface::new(&vk_entry, &vk_instance);
        let vk_surface = Self::create_vulkan_surface_winit(window, &vk_instance, &vk_entry)
            .map_err(|e| {
                error!("Failed to create Vulkan surface: {}", e);
            })
            .ok()?;

        let physical_devices = unsafe { vk_instance.enumerate_physical_devices() }
            .map_err(|e| {
                error!("Failed to enumerate physical devices: {}", e);
            })
            .ok()?;

        let (
            phys_device,
            queue_idx,
            phys_dev_features,
            phys_dev_properties,
            surface_fmt,
            depth_stencil_fmt,
            presentation_mode,
            surface_capabilities,
        ) = pick_physical_device(&vk_instance, vk_surface, &surface_loader, &physical_devices)?;

        let phys_device_mem_props =
            unsafe { vk_instance.get_physical_device_memory_properties(phys_device) };

        info!("Picked device: queue {}, surface format {:?}, depth stencil format {:?}, presentation mode {:?}",
              queue_idx, surface_fmt, depth_stencil_fmt, presentation_mode);
        info!("Surface capabilities: {:?}", surface_capabilities);

        let device_required_extensions = [ash::extensions::khr::Swapchain::name()];

        let graphics_device = std::boxed::Box::pin(
            create_logical_device(
                &vk_instance,
                phys_device,
                queue_idx,
                Some(&device_required_extensions),
            )
            .expect("Failed to create graphics device"),
        );

        info!("Graphics device created successfully.");

        let queue = unsafe { graphics_device.get_device_queue(queue_idx, 0) };

        let swapchain_loader = std::boxed::Box::pin(Swapchain::new(&vk_instance, &graphics_device));
        let (swapchain, max_inflight_frames) = UniqueSwapchain::new(
            &swapchain_loader,
            &surface_capabilities,
            vk_surface,
            surface_fmt,
            presentation_mode,
        )?;

        let swapchain_images =
            unsafe { swapchain_loader.get_swapchain_images(swapchain.swapchain) }
                .map_err(|e| error!("Failed to get images from the swapchain {}", e))
                .ok()?;

        log::info!("Swapchain created, image count {}", max_inflight_frames);

        let cmd_pool = UniqueCommandPool::new(
            &graphics_device,
            &CommandPoolCreateInfo::builder()
                .queue_family_index(queue_idx)
                .flags(
                    CommandPoolCreateFlags::TRANSIENT
                        | CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ),
        )?;

        info!("Command pool created.");

        let renderpass =
            UniqueRenderpass::new(&graphics_device, surface_fmt.format, depth_stencil_fmt)?;

        info!("Renderpass created.");

        let frame_render_data = (0..max_inflight_frames)
            .filter_map(|frame_idx| {
                FrameRenderData::new(
                    &phys_device_mem_props,
                    swapchain_images[frame_idx as usize],
                    &graphics_device,
                    cmd_pool.cmd_pool,
                    depth_stencil_fmt,
                    surface_fmt.format,
                    surface_capabilities.current_extent.width,
                    surface_capabilities.current_extent.height,
                    renderpass.handle,
                )
            })
            .collect::<Vec<_>>();

        if frame_render_data.len() != max_inflight_frames as usize {
            error!(
                "Failed to create frame render data (needed {} frames, got {} frames)",
                max_inflight_frames,
                frame_render_data.len()
            );
        }

        info!("Frame render data created");

        let dpool = DescriptorPoolBuilder::new()
            .add_pool(DescriptorType::SAMPLED_IMAGE, 64)
            .add_pool(DescriptorType::UNIFORM_BUFFER, 64)
            .add_pool(DescriptorType::UNIFORM_BUFFER_DYNAMIC, 64)
            .add_pool(DescriptorType::STORAGE_BUFFER, 64)
            .add_pool(DescriptorType::STORAGE_BUFFER_DYNAMIC, 64)
            .add_pool(DescriptorType::SAMPLER, 64)
            .add_pool(DescriptorType::COMBINED_IMAGE_SAMPLER, 64)
            .build(&graphics_device, 64)?;

        info!("Descriptor pool created");

        let res_loader = ResourceLoader::new(&graphics_device, cmd_pool.cmd_pool)?;

        info!("Resource loading support created");

        let pipeline_cache = UniquePipelineCache::new(&graphics_device)?;

        info!("Pipeline cache created");

        let img_fmt_props = unsafe {
            vk_instance.get_physical_device_image_format_properties(
                phys_device,
                Format::R8G8B8A8_UNORM,
                ImageType::TYPE_2D,
                ImageTiling::OPTIMAL,
                ImageUsageFlags::SAMPLED | ImageUsageFlags::SAMPLED,
                ImageCreateFlags::empty(),
            )
        }
        .expect("Failed to query image format properties ...");

        info!("R8G8B8A8 fmt {:?}", img_fmt_props);

        // img_fmt_props.

        Some(VulkanRenderer {
            work_packages: RefCell::new(Vec::new()),
            pipeline_cache,
            res_loader,
            renderpass,
            current_frame_id: std::cell::Cell::new(0),
            frame_render_data,
            framebuffer_extents: surface_capabilities.current_extent,
            surface_format: surface_fmt.format,
            depth_stencil_format: depth_stencil_fmt,
            cmd_pool,
            descriptor_pool: dpool,
            queue,
            queue_family_index: queue_idx,
            max_inflight_frames,
            swapchain,
            swapchain_loader,
            graphics_device,
            device_memory: phys_device_mem_props,
            device_properties: phys_dev_properties,
            device_features: phys_dev_features,
            debug_utils_msg,
            debug_utils,
            surface_loader,
            vk_surface,
            phys_device,
            vk_instance,
            vk_entry,
        })
    }

    pub fn current_frame_id(&self) -> u32 {
        self.current_frame_id.get()
    }

    pub fn framebuffer_extents(&self) -> Extent2D {
        self.framebuffer_extents
    }

    pub fn max_inflight_frames(&self) -> u32 {
        self.max_inflight_frames
    }

    pub fn graphics_device(&self) -> &Device {
        &self.graphics_device
    }

    pub fn device_memory(&self) -> &PhysicalDeviceMemoryProperties {
        &self.device_memory
    }

    pub fn aligned_size_of_value<T: Sized>(min_align: DeviceSize, data: &T) -> DeviceSize {
        let initial_size = std::mem::size_of_val(data) as DeviceSize;
        if min_align != 0 {
            (initial_size + min_align - 1) & !(min_align - 1)
        } else {
            initial_size
        }
    }

    pub fn aligned_size_of_type<T: Sized>(min_align: DeviceSize) -> DeviceSize {
        let initial_size = std::mem::size_of::<T>() as DeviceSize;
        if min_align != 0 {
            (initial_size + min_align - 1) & !(min_align - 1)
        } else {
            initial_size
        }
    }

    pub fn device_features(&self) -> &PhysicalDeviceFeatures {
        &self.device_features
    }

    pub fn device_properties(&self) -> &PhysicalDeviceProperties {
        &self.device_properties
    }

    pub fn res_loader(&self) -> &ResourceLoader {
        &self.res_loader
    }

    pub fn pipeline_cache(&self) -> PipelineCache {
        self.pipeline_cache.cache
    }

    pub fn descriptor_pool(&self) -> DescriptorPool {
        self.descriptor_pool.dpool
    }

    pub fn begin_resource_loading(&self) {
        unsafe {
            let _ = self.graphics_device.begin_command_buffer(
                self.res_loader.cmd_buf,
                &CommandBufferBeginInfo::builder()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            );
        }
    }

    pub fn wait_resources_loaded(&self) {
        unsafe {
            let _ = self
                .graphics_device
                .end_command_buffer(self.res_loader.cmd_buf);

            let command_buffers = [self.res_loader.cmd_buf];

            let submits = [SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build()];

            let fences = [self.res_loader.fence.fence];

            let _ = self.graphics_device.queue_submit(
                self.queue,
                &submits,
                self.res_loader.fence.fence,
            );

            let _ = self.graphics_device.wait_for_fences(&fences, true, !0u64);
        }
    }

    pub fn viewport(&self) -> Viewport {
        Viewport {
            x: 0f32,
            y: 0f32,
            width: self.framebuffer_extents.width as f32,
            height: self.framebuffer_extents.height as f32,
            min_depth: 1f32,
            max_depth: 0f32,
        }
    }

    pub fn scissor(&self) -> Rect2D {
        Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: Extent2D {
                width: self.framebuffer_extents.width,
                height: self.framebuffer_extents.height,
            },
        }
    }

    fn current_frame_data(&self) -> &FrameRenderData {
        &self.frame_render_data[self.current_frame_id() as usize]
    }

    pub fn begin_frame(&self) {
        let current_frame_data = self.current_frame_data();

        unsafe {
            let fences = [current_frame_data.fence.fence];
            let _ = self.graphics_device.wait_for_fences(&fences, true, !0u64);
        }

        let (available_image, _suboptimal) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                !0u64,
                current_frame_data.sem_img_available.semaphore,
                Fence::null(),
            )
        }
        .expect("Acquire image from swapchain failed");

        //
        //TODO : handle suboptimal

        //
        // wait for submits
        unsafe {
            let fences = [current_frame_data.fence.fence];
            let _ = self.graphics_device.reset_fences(&fences);
        }

        let clear_values = [
            ClearValue {
                color: ClearColorValue {
                    float32: [0f32, 0f32, 0f32, 1f32],
                },
            },
            ClearValue {
                depth_stencil: ClearDepthStencilValue {
                    depth: 1f32,
                    stencil: 0,
                },
            },
        ];

        unsafe {
            let _ = self.graphics_device.reset_command_buffer(
                current_frame_data.command_buffer,
                CommandBufferResetFlags::empty(),
            );

            let _ = self.graphics_device.begin_command_buffer(
                current_frame_data.command_buffer,
                &CommandBufferBeginInfo::builder()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build(),
            );

            self.graphics_device.cmd_begin_render_pass(
                current_frame_data.command_buffer,
                &RenderPassBeginInfo::builder()
                    .render_pass(self.renderpass.handle)
                    .framebuffer(
                        self.frame_render_data[available_image as usize]
                            .framebuffer
                            .framebuffer,
                    )
                    .render_area(Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: self.framebuffer_extents,
                    })
                    .clear_values(&clear_values)
                    .build(),
                SubpassContents::INLINE,
            );
        }
    }

    pub fn end_frame(&self) {
        let current_frame_render_data = self.current_frame_data();

        unsafe {
            self.graphics_device
                .cmd_end_render_pass(current_frame_render_data.command_buffer);
            let _ = self
                .graphics_device
                .end_command_buffer(current_frame_render_data.command_buffer);

            let command_buffers = [current_frame_render_data.command_buffer];
            let wait_semaphores = [current_frame_render_data.sem_img_available.semaphore];
            let signal_semaphores = [current_frame_render_data.sem_img_rending_done.semaphore];
            let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let submits = [SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .wait_semaphores(&wait_semaphores)
                .signal_semaphores(&signal_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .build()];

            let _ = self.graphics_device.queue_submit(
                self.queue,
                &submits,
                current_frame_render_data.fence.fence,
            );

            let swapchains = [self.swapchain.swapchain];
            let swapchain_image_indices = [self.current_frame_id()];

            let suboptimal_swapchain = self
                .swapchain_loader
                .queue_present(
                    self.queue,
                    &PresentInfoKHR::builder()
                        .wait_semaphores(&signal_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&swapchain_image_indices)
                        .build(),
                )
                .map_err(|e| error!("Presentation error {}", e))
                .unwrap_or(true);

            if suboptimal_swapchain {
                info!("Suboptimal present");
            }

            self.current_frame_id
                .set((self.current_frame_id.get() + 1) % self.max_inflight_frames);
        }
    }

    pub fn renderpass(&self) -> RenderPass {
        self.renderpass.handle
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.graphics_device.device_wait_idle();
        }
    }

    pub fn allocate_descriptor_sets(
        &self,
        pipeline: &UniqueGraphicsPipeline,
    ) -> Option<Vec<DescriptorSet>> {
        unsafe {
            self.graphics_device().allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::builder()
                    .set_layouts(pipeline.descriptor_layouts())
                    .descriptor_pool(self.descriptor_pool.dpool)
                    .build(),
            )
        }
        .map_err(|e| error!("Failed to allocate descriptor sets : {}", e))
        .ok()
    }

    pub fn current_command_buffer(&self) -> CommandBuffer {
        self.current_frame_data().command_buffer
    }

    pub fn instance(&self) -> &Instance {
        &self.vk_instance
    }
}

impl std::ops::Drop for VulkanRenderer {
    fn drop(&mut self) {
        self.wait_idle();
    }
}

fn pick_physical_device(
    vk_instance: &Instance,
    vk_surface: SurfaceKHR,
    surface_loader: &Surface,
    physical_devices: &[PhysicalDevice],
) -> Option<(
    PhysicalDevice,
    u32,
    PhysicalDeviceFeatures,
    PhysicalDeviceProperties,
    SurfaceFormatKHR,
    Format,
    ash::vk::PresentModeKHR,
    SurfaceCapabilitiesKHR,
)> {
    let required_extensions = [vk::KhrSwapchainFn::name()];

    for &phys_dev in physical_devices.iter() {
        let device_properties = unsafe { vk_instance.get_physical_device_properties(phys_dev) };

        let device_name =
            unsafe { std::ffi::CStr::from_ptr(device_properties.device_name.as_ptr()) }
                .to_str()
                .unwrap();

        if device_properties.device_type != PhysicalDeviceType::DISCRETE_GPU {
            info!("Rejecting device {} - not a discrete GPU", device_name);
            continue;
        }

        let device_features = unsafe { vk_instance.get_physical_device_features(phys_dev) };
        if device_features.geometry_shader != vk::TRUE
            || device_features.fill_mode_non_solid != vk::TRUE
            || device_features.multi_draw_indirect != vk::TRUE
        {
            info!(
                        "Rejecting device {} because it does not support geometry shaders/wireframe fill/multi draw indirect.",
                        device_name
                    );
            continue;
        }

        let has_required_extensions =
            unsafe { vk_instance.enumerate_device_extension_properties(phys_dev) }
                .map_err(|e| error!("Failed to enumerate extension properties of device {}", e))
                .ok()
                .map(|device_exts| {
                    required_extensions.iter().all(|&required| {
                        device_exts
                            .iter()
                            .find(|dev_ext| unsafe {
                                CStr::from_ptr(dev_ext.extension_name.as_ptr()) == required
                            })
                            .map_or_else(
                                || {
                                    info!(
                                "Rejecting device {} because required extension {} is missing",
                                device_name, ""
                            );
                                    false
                                },
                                |_| true,
                            )
                    })
                })
                .unwrap_or(false);

        if !has_required_extensions {
            continue;
        }

        let depth_stencil_format = find_supported_depth_stencil_format(
            vk_instance,
            phys_dev,
            FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        if depth_stencil_format.is_none() {
            info!(
                "Rejecting device {}, no depth stencil format supported!",
                device_name
            );
            continue;
        }

        let depth_stencil_format = depth_stencil_format.unwrap();

        let queue_family_index =
            find_queue_family_indices(vk_instance, phys_dev, vk_surface, surface_loader);
        if queue_family_index.is_none() {
            info!(
                "Rejecting device {}, no queue family with graphics + presentation support found",
                device_name
            );
            continue;
        }

        let queue_family_index = queue_family_index.unwrap();

        let desired_formats = [
            Format::R8G8B8A8_SRGB,
            Format::B8G8R8A8_SRGB,
            Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8_UNORM,
        ];

        let surface_format =
            unsafe { surface_loader.get_physical_device_surface_formats(phys_dev, vk_surface) }
                .ok()
                .and_then(|surface_formats| {
                    desired_formats
                        .iter()
                        .filter_map(|desired_format| {
                            surface_formats
                                .iter()
                                .find(|surface_fmt| surface_fmt.format == *desired_format)
                        })
                        .nth(0)
                        .copied()
                });

        if surface_format.is_none() {
            info!(
                "Rejecting device {} because it does not support any of {:?}",
                device_name, desired_formats
            );
            continue;
        }

        let surface_format = surface_format.unwrap();

        let desired_present_modes = [
            PresentModeKHR::FIFO,
            PresentModeKHR::MAILBOX,
            PresentModeKHR::IMMEDIATE,
        ];

        let presentation_format = unsafe {
            surface_loader.get_physical_device_surface_present_modes(phys_dev, vk_surface)
        }
        .ok()
        .and_then(|present_modes| {
            desired_present_modes
                .iter()
                .filter_map(|desired_present_mode| {
                    present_modes
                        .iter()
                        .find(|pmode| *pmode == desired_present_mode)
                })
                .nth(0)
                .copied()
        });

        if presentation_format.is_none() {
            info!(
                "Rejecting device {} because it does not support present mode MAILBOX",
                device_name
            );
            continue;
        }

        let presentation_format = presentation_format.unwrap();

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(phys_dev, vk_surface)
                .ok()?
        };

        info!(
            "Surface format {:?}, presentation format {:?}",
            surface_format, presentation_format
        );

        return Some((
            phys_dev,
            queue_family_index,
            device_features,
            device_properties,
            surface_format,
            depth_stencil_format,
            presentation_format,
            surface_caps,
        ));
    }
    None
}

fn find_supported_depth_stencil_format(
    vk_instance: &Instance,
    phys_device: PhysicalDevice,
    features: FormatFeatureFlags,
) -> Option<Format> {
    let depth_stencil_formats = [
        Format::D24_UNORM_S8_UINT,
        Format::D32_SFLOAT_S8_UINT,
        Format::D32_SFLOAT,
    ];

    depth_stencil_formats
        .iter()
        .find(|&&fmt| {
            let format_properties =
                unsafe { vk_instance.get_physical_device_format_properties(phys_device, fmt) };

            format_properties.optimal_tiling_features.contains(features)
        })
        .copied()
}

fn find_queue_family_indices(
    vk_instance: &Instance,
    phys_device: PhysicalDevice,
    vk_surface: SurfaceKHR,
    surface_loader: &Surface,
) -> Option<u32> {
    let queue_family_properties =
        unsafe { vk_instance.get_physical_device_queue_family_properties(phys_device) };

    queue_family_properties.iter().find_map(|qfp| {
        if !qfp
            .queue_flags
            .contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
        {
            return None;
        }

        for qid in 0..qfp.queue_count {
            let queue_supports_present = unsafe {
                surface_loader.get_physical_device_surface_support(phys_device, qid, vk_surface)
            }
            .unwrap_or(false);

            if queue_supports_present {
                return Some(qid);
            }
        }

        None
    })
}

/// Cykaaaaaa
fn create_logical_device(
    vk_instance: &Instance,
    phys_device: PhysicalDevice,
    queue_family_id: u32,
    enabled_exts: Option<&[&CStr]>,
) -> Option<Device> {
    let mut vk11_features = PhysicalDeviceVulkan11Features::builder().build();
    let mut phys_device_features2 =
        PhysicalDeviceFeatures2::builder().push_next(&mut vk11_features);

    unsafe {
        vk_instance.get_physical_device_features2(phys_device, &mut phys_device_features2);
    }

    let queue_priorities = [1f32];
    let enabled_extension_names = enabled_exts
        .map(|ee| {
            ee.iter()
                .map(|e| e.as_ptr() as *const c_char)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![]);

    let queue_create_info = [DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_id)
        .queue_priorities(&queue_priorities)
        .build()];

    unsafe {
        vk_instance.create_device(
            phys_device,
            &DeviceCreateInfo::builder()
                // .enabled_features(phys_device_features)
                .queue_create_infos(&queue_create_info)
                .enabled_extension_names(&enabled_extension_names)
                .push_next(&mut phys_device_features2)
                .build(),
            None,
        )
    }
    .map_err(|e| info!("Failed to create graphics device: {}", e))
    .ok()
}

fn choose_memory_type(
    memory_props: &PhysicalDeviceMemoryProperties,
    memory_requirements: &MemoryRequirements,
    required_flags: MemoryPropertyFlags,
) -> u32 {
    let mut selected_type = !0u32;

    for memory_type in 0u32..32u32 {
        if (memory_requirements.memory_type_bits & (1u32 << memory_type)) != 0
            && memory_props.memory_types[memory_type as usize]
                .property_flags
                .contains(required_flags)
        {
            selected_type = memory_type;
            break;
        }
    }

    selected_type
}

unsafe extern "system" fn debug_message_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    _message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> Bool32 {
    if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(
            "[Vulkan]:: {}",
            CStr::from_ptr((*p_callback_data).p_message)
                .to_str()
                .unwrap_or(r#"cannot display warning"#)
        );
    }

    if message_severity.contains(DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(
            "[Vulkan]:: {}",
            CStr::from_ptr((*p_callback_data).p_message)
                .to_str()
                .unwrap_or(r#"cannot display error"#)
        );
    }
    vk::FALSE
}
