import React, { useState } from "react";
import { styled } from "@mui/material/styles";
import {
  Container,
  Typography,
  Box,
  Button,
  Divider,
  Avatar,
  Card,
  CardContent,
  IconButton,
  Select,
  MenuItem,
  Collapse,
  TextField,
} from "@mui/material";

// import { Navigation, Pagination, Scrollbar, A11y } from "swiper/modules";
// import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/scrollbar";

// import "./../../styles/productDetailsStyles/productImagesCarousel.css";

import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";
import AddShoppingCartIcon from "@mui/icons-material/AddShoppingCart";
import FavoriteIcon from "@mui/icons-material/Favorite";

const reviews = [
  {
    name: "Alexa",
    date: "01-02-2021",
    rating: 5,
    review:
      "I have bought over 10 suits from this seller...wow...what a service...they have size and customize my suits to a perfect and",
  },
  {
    name: "John",
    date: "12-12-2020",
    rating: 4,
    review: "Great quality and fits well. Highly recommend!",
  },
  {
    name: "Emma",
    date: "11-25-2020",
    rating: 3,
    review: "Amazing product and quick delivery!",
  },
  {
    name: "Mike",
    date: "10-05-2020",
    rating: 1,
    review: "Good value for money. Will buy again.",
  },
  {
    name: "Alexa",
    date: "01-02-2021",
    rating: 5,
    review:
      "I have bought over 10 suits from this seller...wow...what a service...they have size and customize my suits to a perfect and",
  },
  {
    name: "John",
    date: "12-12-2020",
    rating: 4,
    review: "Great quality and fits well. Highly recommend!",
  },
  {
    name: "Emma",
    date: "11-25-2020",
    rating: 3,
    review: "Amazing product and quick delivery!",
  },
  {
    name: "Mike",
    date: "10-05-2020",
    rating: 1,
    review: "Good value for money. Will buy again.",
  },
];

const availableSizes = ["S", "M", "L", "XL"];
const availableColors = ["Black", "Red", "Blue", "Green"];

// const productImgs = [
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
//   "/resources/images/m1.jpg",
// ];

export default function ProductDetailsPage() {
  const [selectedSize, setSelectedSize] = useState("L");
  const [selectedColor, setSelectedColor] = useState("Black");
  const [showAllReviews, setShowAllReviews] = useState(false);
  const [userRating, setUserRating] = useState(0);
  const [userReview, setUserReview] = useState("");
  const [userName, setUserName] = useState("");
  const [submittedReviews, setSubmittedReviews] = useState([...reviews]);

  const handleSizeChange = (event) => {
    setSelectedSize(event.target.value);
  };

  const handleColorChange = (event) => {
    setSelectedColor(event.target.value);
  };

  const handleToggleReviews = () => {
    setShowAllReviews(!showAllReviews);
  };

  const handleRating = (index) => {
    setUserRating(index + 1);
  };

  const handleSubmitReview = () => {
    if (userName && userReview && userRating) {
      const newReview = {
        name: userName,
        date: new Date().toLocaleDateString(),
        rating: userRating,
        review: userReview,
      };
      setSubmittedReviews([newReview, ...submittedReviews]);
      setUserName("");
      setUserReview("");
      setUserRating(0);
    }
  };

  const ResponsiveImg = styled("img")(({ theme }) => ({
    width: "60%",
    height: "auto",
    marginBottom: "1rem",
    [theme.breakpoints.down("md")]: {
      width: "80%",
    },
    [theme.breakpoints.down("sm")]: {
      width: "100%",
    },
  }));

  return (
    <main style={{ marginTop: "5rem", marginLeft: "5rem" }}>
      <Container maxWidth="lg">
        <Box textAlign="center">
          <ResponsiveImg src="/resources/images/m1.jpg" alt="Product" />

          {/* <Swiper
            key={productImgs.length} // Add a key to force re-render after all the products have been fetched
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
                slidesPerView: 2,
              },
              420: {
                slidesPerView: 3,
              },
              520: {
                slidesPerView: 4,
              },
              950: {
                slidesPerView: 5,
              },
              1200: {
                slidesPerView: 6,
              },
            }}
          >
            <div className="slide-content">
              {productImgs.map((img, index) => {
                return (
                  <SwiperSlide className="c-card">
                    <div className="image-content">
                      <div className="c-card-image">
                        <img src={img} alt="product" className="c-card-img" />
                      </div>
                    </div>
                  </SwiperSlide>
                );
              })}
            </div>

            <div className="swiper-pagination"></div>
            <div className="swiper-button-next swiper-navBtn"></div>
            <div className="swiper-button-prev swiper-navBtn"></div>
          </Swiper> */}

          <Typography variant="h4" gutterBottom>
            Men Black Hoodie
          </Typography>
          <Typography gutterBottom>
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Similique
            aut atque officiis voluptatem mollitia modi corrupti distinctio
            nihil, velit labore esse facere, itaque eveniet eligendi quidem ad
            eum odit id?
          </Typography>
          <Typography variant="h5" color="error" gutterBottom>
            $39.99
          </Typography>
          <Box display="flex" justifyContent="space-between">
            <Typography variant="body1">Select Size</Typography>
          </Box>
          <Select
            value={selectedSize}
            onChange={handleSizeChange}
            displayEmpty
            inputProps={{ "aria-label": "Select size" }}
            sx={{ marginTop: "1rem", width: "100%" }}
          >
            {availableSizes.map((size) => (
              <MenuItem key={size} value={size}>
                {size}
              </MenuItem>
            ))}
          </Select>

          <Box display="flex" justifyContent="space-between">
            <Typography variant="body1">Select Color</Typography>
          </Box>
          <Select
            value={selectedColor}
            onChange={handleColorChange}
            displayEmpty
            inputProps={{ "aria-label": "Select color" }}
            sx={{ marginTop: "1rem", width: "100%" }}
          >
            {availableColors.map((color) => (
              <MenuItem key={color} value={color}>
                {color}
              </MenuItem>
            ))}
          </Select>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{ marginTop: "1rem" }}
          >
            <IconButton color="primary">
              <AddShoppingCartIcon />
            </IconButton>
            <IconButton color="secondary">
              <FavoriteIcon />
            </IconButton>
          </Box>
        </Box>

        <Divider
          sx={{
            marginY: "2rem",
            borderBottomWidth: "2px",
            borderColor: "primary.main",
          }}
        />

        <Box>
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{ marginBottom: "1rem" }}
          >
            <Typography variant="h6">
              Reviews ({submittedReviews.length})
            </Typography>
            <Button
              variant="text"
              color="primary"
              onClick={handleToggleReviews}
            >
              {showAllReviews ? "Show less reviews" : "See all reviews"}
            </Button>
          </Box>

          <Box sx={{ marginBottom: "1rem" }}>
            <Typography variant="h6">Rate this product</Typography>
            <Box display="flex" alignItems="center">
              {[...Array(5)].map((_, i) => (
                <IconButton key={i} onClick={() => handleRating(i)}>
                  {i < userRating ? (
                    <StarIcon sx={{ color: "yellow" }} />
                  ) : (
                    <StarBorderIcon />
                  )}
                </IconButton>
              ))}
            </Box>
            <TextField
              label="Your Name"
              variant="outlined"
              fullWidth
              sx={{ marginBottom: "1rem" }}
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
            />
            <TextField
              label="Your Review"
              variant="outlined"
              fullWidth
              multiline
              rows={4}
              value={userReview}
              onChange={(e) => setUserReview(e.target.value)}
            />
            <Button
              variant="contained"
              color="primary"
              sx={{ marginTop: "1rem" }}
              onClick={handleSubmitReview}
            >
              Submit
            </Button>
          </Box>

          <Collapse in={!showAllReviews}>
            {submittedReviews.slice(0, 3).map((review, index) => (
              <Card key={index} sx={{ marginBottom: "1rem" }}>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Avatar>{review.name.charAt(0)}</Avatar>
                    <Box sx={{ marginLeft: "1rem" }}>
                      <Typography variant="body1">{review.name}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {review.date}
                      </Typography>
                      <Box display="flex" alignItems="center">
                        {[...Array(5)].map((_, i) =>
                          i < review.rating ? (
                            <StarIcon key={i} sx={{ color: "yellow" }} />
                          ) : (
                            <StarBorderIcon key={i} />
                          )
                        )}
                      </Box>
                    </Box>
                  </Box>
                  <Typography variant="body2" sx={{ marginTop: "1rem" }}>
                    {review.review}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Collapse>

          <Collapse in={showAllReviews}>
            {submittedReviews.map((review, index) => (
              <Card key={index} sx={{ marginBottom: "1rem" }}>
                <CardContent>
                  <Box display="flex" alignItems="center">
                    <Avatar>{review.name.charAt(0)}</Avatar>
                    <Box sx={{ marginLeft: "1rem" }}>
                      <Typography variant="body1">{review.name}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {review.date}
                      </Typography>
                      <Box display="flex" alignItems="center">
                        {[...Array(5)].map((_, i) =>
                          i < review.rating ? (
                            <StarIcon key={i} sx={{ color: "yellow" }} />
                          ) : (
                            <StarBorderIcon key={i} />
                          )
                        )}
                      </Box>
                    </Box>
                  </Box>
                  <Typography variant="body2" sx={{ marginTop: "1rem" }}>
                    {review.review}
                  </Typography>
                </CardContent>
              </Card>
            ))}
          </Collapse>
        </Box>
      </Container>
    </main>
  );
}
