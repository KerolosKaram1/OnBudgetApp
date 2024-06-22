import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Typography, Paper, Grid, IconButton } from "@mui/material";
import { Add, Remove } from "@mui/icons-material";
import FavoriteIcon from "@mui/icons-material/Favorite";
import ShareIcon from "@mui/icons-material/Share";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";

import Alert from "../UtilsComponents/Alert";

export default function CartItem({
  item,
  handleItemRemove,
  setTotalCartItemsPrice,
}) {
  const navigate = useNavigate();

  const [quantity, setQuantity] = useState(1);
  const [inStock, setInStock] = useState(true);
  const [alertFavMessage, setAlertFavMessage] = useState(null);
  const [favorites, setFavorites] = useState({});
  const [isCopied, setIsCopied] = useState(false);

  const stockLimit = 5;
  const alertMsgTime = 1250;

  const handleIncrease = () => {
    if (quantity < stockLimit) {
      setQuantity(quantity + 1);
      setInStock(true);
    } else {
      setInStock(false);
    }
  };

  const handleDecrease = () => {
    if (quantity > 1) {
      setQuantity(quantity - 1);
      setInStock(true);
    } else {
      handleItemRemove(item.id);
    }
  };

  const handleToggleWishlist = (itemID) => {
    setFavorites((prevFavorites) => {
      const updatedFavorites = {
        ...prevFavorites,
        [itemID]: !prevFavorites[itemID],
      };

      if (updatedFavorites[itemID]) {
        setAlertFavMessage("Added to wishlist");
      } else {
        setAlertFavMessage("Removed from wishlist");
      }

      setTimeout(() => {
        setAlertFavMessage(null);
      }, alertMsgTime);

      return updatedFavorites;
    });
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

            {item.price && (
              <Typography
                variant="h6"
                style={{ marginBottom: 10, fontWeight: "bold" }}
              >
                ${item.price.toFixed(2)}
              </Typography>
            )}

            <Typography style={{ marginBottom: 10 }}>Size: M</Typography>
            <Typography style={{ marginBottom: 10 }}>Color: White</Typography>

            <Grid container alignItems="center" style={{ marginBottom: 10 }}>
              <IconButton
                size="small"
                onClick={handleDecrease}
                style={{ color: "red" }}
              >
                <Remove />
              </IconButton>
              <Typography style={{ margin: "0 10px", fontWeight: "bold" }}>
                {quantity}
              </Typography>
              <IconButton
                size="small"
                onClick={handleIncrease}
                style={{ color: "green" }}
              >
                <Add />
              </IconButton>
            </Grid>

            <Typography
              style={{
                color: inStock ? "green" : "red",
                marginBottom: 10,
              }}
            >
              {inStock ? "In Stock" : "Out of Stock"}
            </Typography>
          </Grid>

          <Grid item xs={12} sm={5} style={{ textAlign: "right" }}>
            <Typography
              variant="h6"
              style={{ paddingBottom: "30px", fontWeight: "bold" }}
            >
              Total: ${(item.price * quantity).toFixed(2)}
            </Typography>

            <CustomTooltip title="Add to wishlist" placement="bottom">
              <FavoriteIcon
                aria-label="Favorite"
                sx={{
                  marginRight: 3,
                  cursor: "pointer",
                  color: favorites[item.id] ? "red" : "black",
                  transition: "transform 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.2)",
                  },
                }}
                onClick={() => handleToggleWishlist(item.id)}
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
          </Grid>
        </Grid>
      </Paper>

      {alertFavMessage && (
        <Alert
          message={alertFavMessage}
          msgColor={
            alertFavMessage === "Added to wishlist" ? "#009717" : "#d61919"
          }
        />
      )}

      {isCopied && (
        <Alert message={"Link copied to your clipboard"} msgColor={"#009717"} />
      )}
    </>
  );
}
