import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Typography, Paper, Grid, Button, Box } from "@mui/material";

import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import ShareIcon from "@mui/icons-material/Share";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";

import Alert from "../UtilsComponents/Alert";
import { truncateDescription } from "./../../utils/truncateDescription";

export default function CartItem({ item, handleItemRemove, cartItems }) {
  const alertMsgTime = 1250;
  const navigate = useNavigate();

  // TODO: check if the item is in stock
  // eslint-disable-next-line no-unused-vars
  const [inStock, setInStock] = useState(true);
  // eslint-disable-next-line no-unused-vars
  const [alertAddedToCartMessage, setAlertAddedToCartMessage] = useState(null);
  const [isCopied, setIsCopied] = useState(false);

  const handleToggleCart = (itemID) => {
    // TODO: add and remove from cart
  };

  const RemoveItemsFromWishlist = () => {
    // TODO: remove from wishlist
  };

  const handleShareClick = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setIsCopied(true);

      setTimeout(() => {
        setIsCopied(false);
      }, alertMsgTime);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const CustomTooltip = styled(({ className, ...props }) => (
    <Tooltip {...props} arrow classes={{ popper: className }} />
  ))(({ theme }) => ({
    [`& .${tooltipClasses.tooltip}`]: {
      backgroundColor: "#0194b5",
      color: "#ffffff",
      fontSize: "14px",
      borderRadius: "5px",
    },
    [`& .${tooltipClasses.arrow}`]: {
      color: "#0194b5",
    },
  }));

  return (
    <>
      <Paper elevation={3} style={{ padding: "16px", marginBottom: "16px" }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={2}>
            {item.image && (
              <img
                src={item.image}
                alt={item.title}
                style={{ width: "100%", height: "auto" }}
              />
            )}
          </Grid>
          <Grid item xs={12} sm={5}>
            {item.title && (
              <Typography
                variant="h6"
                sx={{
                  marginBottom: 1,
                  cursor: "pointer",
                  transition: "textDecoration 0.3s ease",
                  "&:hover": {
                    textDecoration: "underline",
                  },
                }}
                onClick={() => navigate("/products/product-details")}
              >
                {item.title}
              </Typography>
            )}

            <Typography style={{}}>
              {truncateDescription(item.description, 15)}
            </Typography>

            {item.price && (
              <Typography
                variant="h6"
                style={{ marginBottom: 10, fontWeight: "bold" }}
              >
                ${item.price.toFixed(2)}
              </Typography>
            )}

            <Typography
              style={{
                color: inStock ? "green" : "red",
                marginBottom: 10,
              }}
            >
              {inStock ? "In Stock" : "Out of Stock"}
            </Typography>

            <Box
              style={{
                display: "flex",
                flexFlow: "row nowrap",
                justifyContent: "space between",
                alignItems: "center",
              }}
            >
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
                onClick={() => RemoveItemsFromWishlist()}
              >
                Remove
              </Button>

              <CustomTooltip title="Add to cart" placement="bottom">
                <ShoppingCartIcon
                  aria-label="Favorite"
                  sx={{
                    marginRight: 3,
                    cursor: "pointer",
                    color: cartItems[item.id] ? "green" : "black",
                    transition: "transform 0.3s ease",
                    "&:hover": {
                      transform: "scale(1.2)",
                    },
                  }}
                  onClick={() => handleToggleCart(item.id)}
                />
              </CustomTooltip>
              <CustomTooltip title="Share" placement="bottom">
                <ShareIcon
                  aria-label="Share"
                  sx={{
                    marginRight: 3,
                    cursor: "pointer",
                    color: "black",
                    transition: "transform 0.3s ease",
                    "&:hover": {
                      transform: "scale(1.2)",
                    },
                  }}
                  onClick={() => handleShareClick()}
                />
              </CustomTooltip>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {alertAddedToCartMessage && (
        <Alert
          message={alertAddedToCartMessage}
          msgColor={
            alertAddedToCartMessage === "Added to cart" ? "#009717" : "#d61919"
          }
        />
      )}

      {isCopied && (
        <Alert message={"Link copied to your clipboard"} msgColor={"#009717"} />
      )}
    </>
  );
}
