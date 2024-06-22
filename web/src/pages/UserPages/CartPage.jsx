import CartItem from "../../components/CartComponents/CartItem";
import TotalAndOrderBtn from "../../components/CartComponents/TotalAndOrderBtn";
import { getCartItems } from "../../utils/getCartItems";
import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Divider,
} from "@mui/material";
import { useEffect, useState } from "react";

export default function CartPage() {
  const [cartItems, setCartItems] = useState([]);
  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    // Cart Items
    const fetchCartItems = async () => {
      try {
        const data = await getCartItems();
        setCartItems(data);
      } catch (error) {
        console.error("Error fetching cart items:", error);
      } finally {
        setIsPageLoading(false);
      }
    };

    fetchCartItems();
  }, []);

  const handleItemRemove = (itemId) => {
    setCartItems(cartItems.filter((item) => item.id !== itemId));
  };

  return (
    // TODO: make it responsive and add marginBottom
    <main
      style={{
        marginTop: "5rem",
        marginLeft: "5rem",
      }}
    >
      <Container maxWidth="lg">
        {/* Page Title */}
        <Typography variant="h4" gutterBottom>
          Shopping Cart
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
              {cartItems.length > 0 ? (
                cartItems.map((item) => (
                  <CartItem
                    key={item.id}
                    item={item}
                    handleItemRemove={handleItemRemove}
                  />
                ))
              ) : (
                // Display this message if there're no cart items
                <Typography variant="h5">Your Cart Is Empty</Typography>
              )}
            </Box>

            {/* Total Items Price And Order Button */}
            {cartItems.length > 0 && <TotalAndOrderBtn cartItems={cartItems} />}
          </>
        )}
      </Container>
    </main>
  );
}
