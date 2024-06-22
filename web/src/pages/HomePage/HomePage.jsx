import { useState, useEffect } from "react";
import ProductsCarousel from "./../../components/HomeComponents/ProductsCarousel";
import CategoriesCarousel from "./../../components/HomeComponents/CategoriesCarousel";
import {
  getRecommendedProducts,
  getBestSellersProducts,
  getWishlistProducts,
} from "./../../utils/getProducts";
import { getAllCategories } from "./../../utils/getAllCategories";

import { CircularProgress, Box } from "@mui/material";

export default function HomePage({ setCartItemsCount }) {
  const [categories, setCategories] = useState([]);
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [bestSellersProducts, setBestSellersProducts] = useState([]);
  const [wishlistProducts, setWishlistProducts] = useState([]);
  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    // Categories
    const fetchCategories = async () => {
      const data = await getAllCategories();
      setCategories(data);
    };

    // Recommended Products
    const fetchRecommendedProducts = async () => {
      const data = await getRecommendedProducts();
      setRecommendedProducts(data);
    };

    // Best Sellers Products
    const fetchBestSellersProducts = async () => {
      const data = await getBestSellersProducts();
      setBestSellersProducts(data.reverse());
    };

    // Wishlist Products
    const fetchWishlistProducts = async () => {
      const data = await getWishlistProducts();
      setWishlistProducts(data);
    };

    Promise.all([
      fetchCategories(),
      fetchRecommendedProducts(),
      fetchBestSellersProducts(),
      fetchWishlistProducts(),
    ])
      .then(() => {
        setIsPageLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching products:", error);
      });
  }, []);

  return (
    <>
      <main
        style={{ marginTop: "5rem", marginLeft: "5rem", paddingBottom: "1rem" }}
      >
        {isPageLoading ? (
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
          >
            <CircularProgress />
          </Box>
        ) : (
          <>
            <CategoriesCarousel
              header={"What are you looking for today? 😁"}
              categories={categories}
            />
            <ProductsCarousel
              header={"Recommended Items 😉"}
              setCartItemsCount={setCartItemsCount}
              recommendedProducts={recommendedProducts}
            />
            <ProductsCarousel
              header={"Best Sellers 😊"}
              setCartItemsCount={setCartItemsCount}
              bestSellersProducts={bestSellersProducts}
            />
            <ProductsCarousel
              header={"Your Wishlist 💖"}
              wishlistProducts={wishlistProducts}
            />
          </>
        )}
      </main>
    </>
  );
}
