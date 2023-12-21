use ash::vk::{
    DescriptorBufferInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout, DescriptorType,
    DeviceSize, PipelineLayout, WriteDescriptorSet,
};

use crate::{
    vk_renderer::{UniqueBuffer, UniqueDescriptorPool, VulkanRenderer},
    ProgramError,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, strum_macros::EnumIter)]
#[repr(u8)]
pub enum BindlessResourceType {
    UniformBuffer,
    Ssbo,
    CombinedImageSampler,
}

impl BindlessResourceType {
    fn as_vk_type(&self) -> ash::vk::DescriptorType {
        match self {
            BindlessResourceType::Ssbo => DescriptorType::STORAGE_BUFFER,
            BindlessResourceType::CombinedImageSampler => DescriptorType::COMBINED_IMAGE_SAMPLER,
            BindlessResourceType::UniformBuffer => DescriptorType::UNIFORM_BUFFER,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BindlessResourceHandle2(u32);

impl BindlessResourceHandle2 {
    pub fn get_type(&self) -> BindlessResourceType {
        let bits = self.0 & 0b11;
        match bits {
            1 => BindlessResourceType::Ssbo,
            2 => BindlessResourceType::CombinedImageSampler,
            0 => BindlessResourceType::UniformBuffer,
            _ => todo!("Handle this case"),
        }
    }

    pub fn handle(&self) -> u32 {
        self.0 >> 2
    }

    pub fn new(ty: BindlessResourceType, id: u32) -> Self {
        match ty {
            BindlessResourceType::Ssbo => Self(id << 2),
            BindlessResourceType::CombinedImageSampler => Self(1 | (id << 2)),
            BindlessResourceType::UniformBuffer => Self(2 | (id << 2)),
        }
    }
}

pub struct BindlessResourceSystem {
    dpool: UniqueDescriptorPool,
    set_layouts: Vec<DescriptorSetLayout>,
    descriptor_sets: Vec<DescriptorSet>,
    ssbos: Vec<BindlessResourceHandle2>,
    samplers: Vec<BindlessResourceHandle2>,
    ubos: Vec<BindlessResourceHandle2>,
    bindless_pipeline_layout: PipelineLayout,
}

impl BindlessResourceSystem {
    pub fn descriptor_sets(&self) -> &[DescriptorSet] {
        &self.descriptor_sets
    }
    pub fn bindless_pipeline_layout(&self) -> PipelineLayout {
        self.bindless_pipeline_layout
    }

    pub fn new(vks: &VulkanRenderer) -> Result<Self, ProgramError> {
        let dpool_sizes = [
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1024),
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1024),
            *DescriptorPoolSize::builder()
                .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1024),
        ];

        let dpool = UniqueDescriptorPool::new(vks, &dpool_sizes, 8)?;

        use strum::IntoEnumIterator;
        let set_layouts = BindlessResourceType::iter()
            .map(|res_type| {
                unsafe {
                    let mut flag_info =
                        *ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                            .binding_flags(&[ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND]);

                    let dcount = match res_type {
                        BindlessResourceType::UniformBuffer => 8,
                        _ => 1024,
                    };

                    vks.graphics_device().create_descriptor_set_layout(
                        &ash::vk::DescriptorSetLayoutCreateInfo::builder()
                            .push_next(&mut flag_info)
                            .flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                            .bindings(&[*ash::vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(dcount)
                                .binding(0)
                                .descriptor_type(res_type.as_vk_type())
                                .stage_flags(ash::vk::ShaderStageFlags::ALL)]),
                        None,
                    )
                }
                .expect("Failed to create layout")
            })
            .collect::<Vec<_>>();

        let descriptor_sets = unsafe {
            vks.graphics_device().allocate_descriptor_sets(
                &ash::vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(dpool.dpool)
                    .set_layouts(&set_layouts),
            )
        }?;

        log::info!(
            "Allocated {} descriptor sets for bindless pipelines",
            descriptor_sets.len()
        );

        let bindless_pipeline_layout = unsafe {
            vks.graphics_device().create_pipeline_layout(
                &ash::vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&[*ash::vk::PushConstantRange::builder()
                        .stage_flags(ash::vk::ShaderStageFlags::ALL)
                        .offset(0)
                        .size(std::mem::size_of::<u32>() as _)]),
                None,
            )
        }?;

        Ok(Self {
            dpool,
            set_layouts,
            descriptor_sets,
            ssbos: vec![],
            samplers: vec![],
            ubos: vec![],
            bindless_pipeline_layout,
        })
    }

    // TODO: try to make the functions below a bit more generic maybe ??!!
    // code is pretty similar

    pub fn register_ssbo(
        &mut self,
        vks: &VulkanRenderer,
        ssbo: &UniqueBuffer,
    ) -> BindlessResourceHandle2 {
        let idx = self.ssbos.len() as u32;
        let handle = BindlessResourceHandle2::new(BindlessResourceType::Ssbo, idx);
        self.ssbos.push(handle);

        unsafe {
            let buf_info = *DescriptorBufferInfo::builder()
                .buffer(ssbo.buffer)
                .range(ash::vk::WHOLE_SIZE)
                .offset(0);
            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::Ssbo as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            vks.graphics_device()
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }

    fn register_chunked_buffer(
        vks: &VulkanRenderer,
        buff: &UniqueBuffer,
        count: usize,
        bindless_handles: &mut Vec<BindlessResourceHandle2>,
        desc_set: DescriptorSet,
        ty: BindlessResourceType,
    ) -> Vec<BindlessResourceHandle2> {
        let mut handles = vec![];
        let mut buffer_info = vec![];
        let mut buffer_writes = vec![];

        for i in 0..count {
            let idx = bindless_handles.len() as u32;
            let handle = BindlessResourceHandle2::new(ty, idx);
            bindless_handles.push(handle);
            handles.push(handle);

            buffer_info.push(
                *DescriptorBufferInfo::builder()
                    .buffer(buff.buffer)
                    .range(buff.aligned_slab_size as DeviceSize)
                    .offset(i as DeviceSize * buff.aligned_slab_size as DeviceSize),
            );

            buffer_writes.push(
                *WriteDescriptorSet::builder()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(idx)
                    .descriptor_type(ty.as_vk_type())
                    .buffer_info(std::slice::from_ref(&buffer_info[i])),
            );
        }

        unsafe {
            vks.graphics_device()
                .update_descriptor_sets(&buffer_writes, &[]);
        }

        handles
    }

    pub fn register_chunked_uniform(
        &mut self,
        vks: &VulkanRenderer,
        ubo: &UniqueBuffer,
        count: usize,
    ) -> Vec<BindlessResourceHandle2> {
        Self::register_chunked_buffer(
            vks,
            ubo,
            count,
            &mut self.ubos,
            self.descriptor_sets[BindlessResourceType::UniformBuffer as usize],
            BindlessResourceType::UniformBuffer,
        )
    }

    pub fn register_chunked_ssbo(
        &mut self,
        vks: &VulkanRenderer,
        ssbo: &UniqueBuffer,
        count: usize,
    ) -> Vec<BindlessResourceHandle2> {
        Self::register_chunked_buffer(
            vks,
            ssbo,
            count,
            &mut self.ssbos,
            self.descriptor_sets[BindlessResourceType::Ssbo as usize],
            BindlessResourceType::Ssbo,
        )
    }

    pub fn register_image(
        &mut self,
        vks: &VulkanRenderer,
        imgview: ash::vk::ImageView,
        sampler: ash::vk::Sampler,
    ) -> BindlessResourceHandle2 {
        let idx = self.samplers.len() as u32;
        let handle = BindlessResourceHandle2::new(BindlessResourceType::CombinedImageSampler, idx);
        self.samplers.push(handle);

        unsafe {
            let img_info = *ash::vk::DescriptorImageInfo::builder()
                .image_view(imgview)
                .sampler(sampler)
                .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::CombinedImageSampler as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&img_info));

            vks.graphics_device()
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }

    pub fn register_uniform_buffer(
        &mut self,
        vks: &VulkanRenderer,
        ubo: &UniqueBuffer,
    ) -> BindlessResourceHandle2 {
        let idx = self.ubos.len() as u32;
        let handle = BindlessResourceHandle2::new(BindlessResourceType::UniformBuffer, idx);
        self.ubos.push(handle);

        unsafe {
            let buf_info = *DescriptorBufferInfo::builder()
                .buffer(ubo.buffer)
                .range(ash::vk::WHOLE_SIZE)
                .offset(0);

            let write = *WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[BindlessResourceType::UniformBuffer as usize])
                .dst_binding(0)
                .dst_array_element(idx)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            vks.graphics_device()
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }

        handle
    }
}
