import { Typography, Box, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";

export default function TotalAndOrderBtn({ cartItems }) {
  const totalPrice = cartItems.reduce((sum, item) => sum + item.price, 0);
  const navigate = useNavigate();

  return (
    <>
      <Box
        display="flex"
        justifyContent="space-between"
        alignItems="center"
        mt={4}
        style={{ paddingBottom: "2rem" }}
      >
        <Typography variant="h5">Total: ${totalPrice.toFixed(2)}</Typography>
        <Button
          variant="contained"
          color="primary"
          sx={{ textTransform: "none" }}
          onClick={() => navigate("/user/cart/checkout")}
        >
          Proceed to checkout
        </Button>
      </Box>
    </>
  );
}
