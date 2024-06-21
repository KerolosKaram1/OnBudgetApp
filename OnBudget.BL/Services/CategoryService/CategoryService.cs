﻿using OnBudget.BL.DTOs.CategoryDtos;
using OnBudget.BL.DTOs.PictureDtos;
using OnBudget.BL.DTOs.ProductDtos;
using OnBudget.DA.Model.Entities;
using OnBudget.DA.Repository.CategoryRepo;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnBudget.BL.Services.CategoryService
{
    public class CategoryService : ICategoryService
    {
        private readonly ICategoryRepository _categoryRepository;

        public CategoryService(ICategoryRepository categoryRepository)
        {
            _categoryRepository = categoryRepository;
        }

        public async Task<ReadCategoryDto> GetCategoryByNameAsync(string name)
        {
            var category = await _categoryRepository.GetByNameAsync(name);
            return MapToDto(category);
        }

        //public async Task<IEnumerable<ReadCategoryDto>> GetAllCategoriesAsync()
        //{
        //    var categories = await _categoryRepository.GetAllAsync();
        //    return categories.Select(MapToDto);
        //}

        public async Task<string> AddCategoryAsync(WriteCategoryDto writeCategoryDto)
        {
            var category = new Category
            {
                Name = writeCategoryDto.Name,
                //Description = writeCategoryDto.Description
            };
            await _categoryRepository.AddAsync(category);
            return category.Name;
        }

        public async Task UpdateCategoryAsync(string name, WriteCategoryDto writeCategoryDto)
        {
            var category = await _categoryRepository.GetByNameAsync(name);
            if (category != null)
            {
                category.Name = writeCategoryDto.Name;
                //category.Description = writeCategoryDto.Description;
                await _categoryRepository.UpdateAsync(category);
            }
        }

        public async Task RemoveCategoryAsync(string cname)
        {
            await _categoryRepository.RemoveAsync(cname);
        }

        private static ReadCategoryDto MapToDto(Category category)
        {
            return new ReadCategoryDto
            {
                //Id = category.Id,
                Name = category.Name,
                //Description = category.Description
                Products = category.Products.Select(product => new ReadProductDto
                {
                    Id = product.Id,
                    ProductName = product.ProductName,
                    ProductDescription = product.ProductDescription,
                    UnitPrice = product.UnitPrice,
                    Color = product.Color,
                    CategoryName = product.CategoryName,
                    SupplierHandle = product.SupplierHandle,
                    Pictures = product.Pictures.Select(Picture => new ReadPictureDto
                    {
                        Front = Picture.Front,
                        Back = Picture.Back,

                    }).ToList()
                }).ToList()
            };
        }
        public async Task<List<ReadCategoryDto>> GetAllCategoriesWithProductsAsync()
        {
            var categories = await _categoryRepository.GetAllCategoriesWithProductsAsync();
            return categories.Select(category => new ReadCategoryDto
            {
                //Id = category.Id,
                Name = category.Name,
                //Description = category.Description,
                Products = category.Products.Select(product => new ReadProductDto
                {
                    Id = product.Id,
                    ProductName = product.ProductName,
                    ProductDescription = product.ProductDescription,
                    UnitPrice = product.UnitPrice,
                    Color = product.Color,
                    CategoryName = product.CategoryName,
                    SupplierHandle = product.SupplierHandle,
                    Pictures = product.Pictures.Select(Picture => new ReadPictureDto
                    {
                        Front = Picture.Front,
                        Back = Picture.Back,

                    }).ToList()
                }).ToList()
            }).ToList();
        }
        }
}