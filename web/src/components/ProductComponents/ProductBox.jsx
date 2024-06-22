import { useNavigate } from "react-router-dom";

import { Box } from "@mui/material";

import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";

import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import FavoriteIcon from "@mui/icons-material/Favorite";
import ShareIcon from "@mui/icons-material/Share";

import { truncateDescription } from "./../../utils/truncateDescription";

// import Alert from "../UtilsComponents/Alert";

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

export default function ProductBox({
  product,
  handleAddingToWishlist,
  handleAddingToCart,
  handleShareClick,
}) {
  // TODO: get discountPrice from the database
  const discountPrice = 50; // Example 10% discount
  const navigate = useNavigate();

  return (
    <>
      <Box className="card">
        <div className="image-content">
          <span className="overlay"></span>
          <div className="card-image">
            <img src={product.image} alt="product" className="card-img" />
          </div>
        </div>
        <div className="card-content">
          <h2 className="name">{product.title}</h2>
          <p className="description">
            {truncateDescription(product.description, 15)}
          </p>
          <span>
            Price:{" "}
            {product.price > 50 ? (
              <>
                <strike>${product.price}</strike> <span>${discountPrice}</span>
              </>
            ) : (
              <span>${product.price}</span>
            )}
          </span>
          <div className="buttons-container">
            <div className="icons-container">
              <CustomTooltip title="Add to cart" placement="bottom">
                <ShoppingCartIcon
                  className="icon"
                  // sx={{ color: cart[index] ? "#009717" : "inherit" }}
                  // onClick={() => handleAddingToCart()}
                />
              </CustomTooltip>
              <CustomTooltip title="Add to wishlist" placement="bottom">
                <FavoriteIcon
                  className="icon"
                  // sx={{ color: favorites[index] ? "#d61919" : "inherit" }}
                  // onClick={() => handleAddingToWishlist()}
                />
              </CustomTooltip>

              <CustomTooltip title="Share" placement="bottom">
                <ShareIcon
                  className="icon"
                  // onClick={handleShareClick}
                />
              </CustomTooltip>
            </div>
            <CustomTooltip title="Product details" placement="bottom">
              <button
                className="button"
                onClick={() => navigate("/products/product-details")}
              >
                More
              </button>
            </CustomTooltip>
          </div>
        </div>
      </Box>

      {/* {alertCartMessage && (
        <Alert
          message={alertCartMessage}
          msgColor={
            alertCartMessage === "Added to cart" ? "#009717" : "#d61919"
          }
        />
      )} */}
    </>
  );
}
