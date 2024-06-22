import { useState, useEffect } from "react";
import { getAllUserCoupons } from "./../../utils/getCoupons";
import { Box, CircularProgress } from "@mui/material";

export default function CouponsPage() {
  const [userCoupons, setUserCoupons] = useState([]);
  const [isPageLoading, setIsPageLoading] = useState(true);

  useEffect(() => {
    const fetchCoupons = async () => {
      try {
        const data = await getAllUserCoupons();
        setUserCoupons(data);
      } catch (error) {
        console.error("Error fetching cart items:", error);
      } finally {
        setIsPageLoading(false);
      }
    };

    fetchCoupons();
  }, []);

  return (
    <main
      style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
      }}
    >
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
      ) : userCoupons.length > 0 ? (
        userCoupons.map((coupon, index) => {
          return (
            <div key={index}>
              <h2>{coupon.name}</h2>
              <p>Discount: {coupon.discount}%</p>
              <p>Valid until: {coupon.expirationDate}</p>
            </div>
          );
        })
      ) : (
        <h1>
          Sorry, you have no coupons 😥 <br /> Try to get some.
        </h1>
      )}
    </main>
  );
}
