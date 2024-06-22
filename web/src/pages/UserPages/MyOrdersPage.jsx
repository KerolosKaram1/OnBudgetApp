import Order from "../../components/MyOrdersComponents/Order";

import { useState, useEffect } from "react";
import { getAllOrders } from "./../../utils/getOrders";

import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Divider,
} from "@mui/material";

export default function MyOrdersPage() {
  const [isPageLoading, setIsPageLoading] = useState(true);
  const [allOrders, setAllOrders] = useState([]);
  const [alertOrderCanceledMessage, setAlertOrderCanceledMessage] =
    useState(null);
  const alertMsgTime = 1250;

  const handleOrderCanceling = (orderNum) => {
    // TODO: instead of deletign the order change its status
    setAllOrders(allOrders.filter((order) => order.orderNum !== orderNum));
    setAlertOrderCanceledMessage("Order Canceled");

    setTimeout(() => {
      setAlertOrderCanceledMessage(null);
    }, alertMsgTime);
  };

  useEffect(() => {
    // All Orders
    const fetchOrders = async () => {
      try {
        const data = await getAllOrders();
        setAllOrders(data);
      } catch (error) {
        console.error("Error fetching orders items:", error);
      } finally {
        setIsPageLoading(false);
      }
    };

    fetchOrders();
  }, []);

  return (
    <main
      style={{ marginTop: "5rem", marginLeft: "5rem", marginBottom: "5rem" }}
    >
      <Container maxWidth="lg">
        {/* Page Title */}
        <Typography variant="h4" gutterBottom>
          My Orders 😃
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
              {allOrders.length > 0 ? (
                allOrders.map((order) => (
                  <Order
                    key={order.orderNum}
                    order={order}
                    handleOrderCanceling={handleOrderCanceling}
                    alertOrderCanceledMessage={alertOrderCanceledMessage}
                  />
                ))
              ) : (
                // Display this message if there're no cart items
                <Typography variant="h5">You Have no orders!😥</Typography>
              )}
            </Box>
          </>
        )}
      </Container>
    </main>
  );
}
