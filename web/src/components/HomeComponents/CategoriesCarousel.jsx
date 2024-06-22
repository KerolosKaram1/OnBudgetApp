import { useNavigate } from "react-router-dom";

// import utility functions

// import Swiper core and required modules
import { Navigation, Pagination, Scrollbar, A11y } from "swiper/modules";
import { Swiper, SwiperSlide } from "swiper/react";

// import swiper styles
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/scrollbar";

// import mui components
// import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
// import { styled } from "@mui/material/styles";

// import custom swiper styles
import "./../../styles/homePageStyles/CategoriesCarousel.css";

// const CustomTooltip = styled(({ className, ...props }) => (
//   <Tooltip {...props} arrow classes={{ popper: className }} />
// ))(({ theme }) => ({
//   [`& .${tooltipClasses.tooltip}`]: {
//     backgroundColor: "#0194b5",
//     color: "#ffffff",
//     fontSize: "14px",
//     borderRadius: "5px",
//   },
//   [`& .${tooltipClasses.arrow}`]: {
//     color: "#0194b5",
//   },
// }));

export default function ProductsCarousel({ header, categories }) {
  const navigate = useNavigate();

  return (
    <>
      <h1
        className="header"
        onClick={() => {
          if (header === "What are you looking for today? 😁") {
            navigate("/categories");
          }
        }}
      >
        {header}
      </h1>
      <Swiper
        key={categories.length} // Add a key to force re-render after all the products have been fetched
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
        // onSlideChange={() => console.log("hahaa slider has been changed")}
      >
        <div className="slide-content">
          {categories.map((product, index) => {
            return (
              <SwiperSlide
                className="c-card"
                onClick={() => {
                  navigate(`/categories/${product.title}`);
                }}
              >
                {/* <CustomTooltip
                  title={product.title}
                  placement="bottom"
                  key={index}
                > */}
                <div className="image-content">
                  <div className="c-card-image">
                    <img
                      src={product.image}
                      alt="product"
                      className="c-card-img"
                    />
                  </div>
                </div>

                <div className="card-content">
                  <span className="c-name">{product.title}</span>
                </div>
                {/* </CustomTooltip> */}
              </SwiperSlide>
            );
          })}
        </div>

        <div className="swiper-pagination"></div>
        <div className="swiper-button-next swiper-navBtn"></div>
        <div className="swiper-button-prev swiper-navBtn"></div>
      </Swiper>
    </>
  );
}
