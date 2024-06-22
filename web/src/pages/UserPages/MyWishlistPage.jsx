import WishlistItem from "../../components/WishlistComponents/WishlistItem";

import { getWishlistItems } from "../../utils/getWishlistItems";
import { getCartItems } from "../../utils/getCartItems";
import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Divider,
} from "@mui/material";
import { useEffect, useState } from "react";

export default function ProductsPage() {
  const [wishlistItems, setWishlistItems] = useState([]);
  const [cartItems, setCartItems] = useState([]);
  const [isPageLoading, setIsPageLoading] = useState(true);

  const handleItemRemove = (itemId) => {
    setWishlistItems(wishlistItems.filter((item) => item.id !== itemId));
  };

  useEffect(() => {
    // Wishlist Items
    const fetchWishlistItems = async () => {
      try {
        const data = await getWishlistItems();
        setWishlistItems(data);
      } catch (error) {
        console.error("Error fetching wishlist items:", error);
      } finally {
        setIsPageLoading(false);
      }
    };

    // Cart Items
    const fetchCartItems = async () => {
      try {
        const data = await getCartItems();
        setCartItems(data);
      } catch (error) {
        console.error("Error fetching cart items:", error);
      }
    };

    fetchCartItems();
    fetchWishlistItems();
  }, []);

  return (
    <main
      style={{ marginTop: "5rem", marginLeft: "5rem", marginBottom: "5rem" }}
    >
      <Container maxWidth="lg">
        {/* Page Title */}
        <Typography variant="h4" gutterBottom>
          My Wishlist 😍
        </Typography>

        <Divider sx={{ marginBottom: "2rem" }} />

        {/* Display loading indicator if fetching data */}
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
            <Box marginTop={2}>
              {/* Cart items if they're exist */}
              {wishlistItems.length > 0 ? (
                wishlistItems.map((item) => (
                  <WishlistItem
                    key={item.id}
                    item={item}
                    handleItemRemove={handleItemRemove}
                    cartItems={cartItems}
                  />
                ))
              ) : (
                // Display this message if there're no cart items
                <Typography variant="h5">Your Wishlist Is Empty</Typography>
              )}
            </Box>
          </>
        )}
      </Container>
    </main>
  );
}
