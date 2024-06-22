import { useState, useEffect } from "react";
import { getAllProductsInThisCategory } from "../../utils/getProducts";

import ProductBox from "./../../components/ProductComponents/ProductBox";
import FilterSection from "./../../components/ProductComponents/FilterSection";

import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Divider,
  Grid,
  Button,
} from "@mui/material";

export default function ProductsPage({ pageTitle }) {
  const [categoryProducts, setCategoryProducts] = useState([]);
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterCriteria, setFilterCriteria] = useState("");
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");

  useEffect(() => {
    // Categories
    const fetchCategories = async () => {
      const data = await getAllProductsInThisCategory(pageTitle);
      setCategoryProducts(data);
      setIsPageLoading(false);
    };

    fetchCategories();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFilterChange = (event) => {
    setFilterCriteria(event.target.value);
    // Apply other filtering logic here if needed
  };

  const handleMinPriceChange = (event) => {
    setMinPrice(event.target.value);
  };

  const handleMaxPriceChange = (event) => {
    setMaxPrice(event.target.value);
  };

  const applyFilters = (products) => {
    return products.filter((product) => {
      const meetsPriceCriteria =
        (minPrice === "" || product.price >= minPrice) &&
        (maxPrice === "" || product.price <= maxPrice);

      // Add more filter criteria as needed
      return meetsPriceCriteria;
    });
  };

  const filteredProducts = applyFilters(categoryProducts);

  return (
    <main style={{ marginTop: "5rem", marginLeft: "5rem" }}>
      <Container maxWidth="lg" style={{ paddingBottom: "1rem" }}>
        {/* Page Title */}
        <Typography variant="h4" gutterBottom>
          {pageTitle}
        </Typography>

        {/* Filter Section */}
        <Button
          onClick={() => setFilterOpen(!filterOpen)}
          aria-expanded={filterOpen}
        >
          {filterOpen ? "Hide Filters" : "Show Filters"}
        </Button>

        <FilterSection
          filterOpen={filterOpen}
          filterCriteria={filterCriteria}
          handleFilterChange={handleFilterChange}
          minPrice={minPrice}
          maxPrice={maxPrice}
          handleMaxPriceChange={handleMaxPriceChange}
          handleMinPriceChange={handleMinPriceChange}
        />

        <Divider sx={{ marginBottom: "2rem" }} />

        {/* Products Grid Container */}
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
            <Grid container spacing={4}>
              {/* Cart items if they're exist */}
              {filteredProducts.length > 0 ? (
                filteredProducts.map((product) => (
                  <Grid
                    item
                    xs={12}
                    sm={6}
                    md={4}
                    lg={3}
                    key={product.id}
                    style={{ marginBottom: "1rem" }}
                  >
                    <ProductBox
                      key={product.id}
                      product={product}
                      // handleAddingToCart={handleAddingToCart}
                      // handleAddingToWishlist={handleAddingToWishlist}
                      // handleShareClick={handleShareClick}
                    />
                  </Grid>
                ))
              ) : (
                // Display this message if there're no cart items
                <Typography variant="h5">
                  There're No Products In This Category
                </Typography>
              )}
            </Grid>
          </>
        )}
      </Container>
    </main>
  );
}
