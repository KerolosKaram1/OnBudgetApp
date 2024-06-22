// import from react
import { useState } from "react";
import { useNavigate } from "react-router-dom";

// import mui icons
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
// import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder";
import FavoriteIcon from "@mui/icons-material/Favorite";
import ShareIcon from "@mui/icons-material/Share";

// import utility functions
import { truncateDescription } from "./../../utils/truncateDescription";

// import Swiper core and required modules
import { Navigation, Pagination, Scrollbar, A11y } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";

// import swiper styles
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/scrollbar";

// import mui components
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";

// import custom swiper styles
import "./../../styles/homePageStyles/ProductsCarousel.css";

// import other components
import Alert from "../UtilsComponents/Alert";

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

export default function ProductsCarousel({
  header,
  setCartItemsCount,
  recommendedProducts,
  bestSellersProducts,
  wishlistProducts,
}) {
  const alertMsgTime = 1250;
  const [isCopied, setIsCopied] = useState(false);
  const [favorites, setFavorites] = useState({});
  const [cart, setCart] = useState({});
  const [alertFavMessage, setAlertFavMessage] = useState(null);
  const [alertCartMessage, setAlertCartMessage] = useState(null);
  const navigate = useNavigate();

  // TODO: get discountPrice from the database
  const discountPrice = 50; // Example 10% discount

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

  const handleFavoriteClick = (index) => {
    setFavorites((prevFavorites) => {
      const updatedFavorites = {
        ...prevFavorites,
        [index]: !prevFavorites[index],
      };

      if (updatedFavorites[index]) {
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

  const handleCartClick = (index) => {
    setCart((prevCart) => {
      const updatedCart = {
        ...prevCart,
        [index]: !prevCart[index],
      };

      if (updatedCart[index]) {
        setAlertCartMessage("Added to cart");
      } else {
        setAlertCartMessage("Removed from cart");
      }

      setTimeout(() => {
        setAlertCartMessage(null);
      }, alertMsgTime);

      setCartItemsCount(Object.values(updatedCart).filter(Boolean).length);
      return updatedCart;
    });
  };

  // A Swiper for each section
  function renderSwiperProducts(products) {
    return products.map((product, index) => (
      <SwiperSlide className="card" key={index}>
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
                  sx={{ color: cart[index] ? "#009717" : "inherit" }}
                  onClick={() => handleCartClick(index)}
                />
              </CustomTooltip>
              <CustomTooltip title="Add to wishlist" placement="bottom">
                <FavoriteIcon
                  className="icon"
                  sx={{ color: favorites[index] ? "#d61919" : "inherit" }}
                  onClick={() => handleFavoriteClick(index)}
                />
              </CustomTooltip>

              <CustomTooltip title="Share" placement="bottom">
                <ShareIcon className="icon" onClick={handleShareClick} />
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
      </SwiperSlide>
    ));
  }

  return (
    <>
      <h1
        className="header"
        onClick={() => {
          if (header === "Recommended Items 😉") {
            navigate("/products/rec-items");
          } else if (header === "Best Sellers 😊") {
            navigate("/products/best-sellers");
          } else if (header === "Your Wishlist 💖") {
            navigate("/user/wishlist");
          }
        }}
      >
        {header}
      </h1>
      <Swiper
        // key={recommendedProducts.length} // Add a key to force re-render after all the products have been fetched
        className="slide-container"
        modules={[Navigation, Pagination, Scrollbar, A11y]}
        slidesPerView={3}
        spaceBetween={25}
        loop={true}
        effect="coverflow"
        grabCursor={true}
        navigation={{
          nextEl: ".swiper-button-next",
          prevEl: ".swiper-button-prev",
        }}
        pagination={{
          el: ".swiper-pagination",
          clickable: true,
          dynamicBullets: true,
        }}
        breakpoints={{
          0: {
            slidesPerView: 1,
          },
          520: {
            slidesPerView: 2,
          },
          950: {
            slidesPerView: 3,
          },
          1200: {
            slidesPerView: 4,
          },
        }}
        // onSlideChange={() => console.log("hahaa slider has been changed")}
      >
        <div className="slide-content">
          {header === "Recommended Items 😉" &&
            renderSwiperProducts(recommendedProducts)}
          {header === "Best Sellers 😊" &&
            renderSwiperProducts(bestSellersProducts)}
          {header === "Your Wishlist 💖" &&
            renderSwiperProducts(wishlistProducts)}
        </div>

        <div className="swiper-pagination"></div>
        <div className="swiper-button-next swiper-navBtn"></div>
        <div className="swiper-button-prev swiper-navBtn"></div>
      </Swiper>

      {isCopied && (
        <Alert message={"Link copied to your clipboard"} msgColor={"#009717"} />
      )}

      {alertFavMessage && (
        <Alert
          message={alertFavMessage}
          msgColor={
            alertFavMessage === "Added to wishlist" ? "#009717" : "#d61919"
          }
        />
      )}

      {alertCartMessage && (
        <Alert
          message={alertCartMessage}
          msgColor={
            alertCartMessage === "Added to cart" ? "#009717" : "#d61919"
          }
        />
      )}
    </>
  );
}
