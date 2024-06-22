import { Typography, Paper, Grid, Button, Box } from "@mui/material";

import Alert from "../UtilsComponents/Alert";

export default function CartItem({
  order,
  handleOrderCanceling,
  alertOrderCanceledMessage,
}) {
  return (
    <>
      <Paper elevation={3} style={{ padding: "16px", marginBottom: "16px" }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={5}>
            {order.orderNum && (
              <Typography
                variant="h6"
                sx={{
                  marginBottom: 1,
                  fontWeight: "bold",
                }}
              >
                Order Number: {order.orderNum}
              </Typography>
            )}

            {order.total && (
              <Typography variant="h6" style={{ marginBottom: 10 }}>
                Total: ${order.total.toFixed(2)}
              </Typography>
            )}

            <Typography
              style={{
                color:
                  order.status === "canceled" ||
                  order.status === "canceled and returned"
                    ? "red"
                    : "green",
                marginBottom: 10,
              }}
            >
              Order Status: {order.status}
            </Typography>

            <Box
              style={{
                display: "flex",
                flexFlow: "row nowrap",
                justifyContent: "space between",
                alignItems: "center",
              }}
            >
              {order.status !== "canceled" &&
                order.status !== "canceled and returned" &&
                order.status !== "delivered" && (
                  <Button
                    variant="contained"
                    style={{ backgroundColor: "red" }}
                    size="small"
                    sx={{
                      fontSize: { xs: "0.7rem", sm: "0.75rem" }, // Adjust button text size
                      textTransform: "none",
                      marginRight: 3,
                      cursor: "pointer",
                      transition: "transform 0.3s ease",
                      "&:hover": {
                        transform: "scale(1.05)",
                      },
                    }}
                    onClick={() => handleOrderCanceling(order.orderNum)}
                  >
                    Cancel
                  </Button>
                )}
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {alertOrderCanceledMessage && (
        <Alert
          message={alertOrderCanceledMessage}
          msgColor={
            alertOrderCanceledMessage === "Order Canceled"
              ? "#d61919"
              : "#009717"
          }
        />
      )}
    </>
  );
}
