import { BrowserRouter, Route, Routes } from "react-router-dom";
import { useState, useEffect } from "react";

import { getAllCategories } from "./utils/getAllCategories";
import { getUnreadNotifications } from "./utils/getNotifications";

import ProductsPage from "./pages/ProductPages/ProductsPage.jsx";
import ProductDetailsPage from "./pages/ProductPages/ProductDetailsPage.jsx";
import CategoriesPage from "./pages/ProductPages/CategoriesPage.jsx";
import ChatbotAIAndImgCreatorPage from "./pages/ChatbotAndImageCreatorPages/ChatbotAIAndImgCreatorPage";
import ChatbotPage from "./pages/ChatbotAndImageCreatorPages/ChatbotPage";
import ImageCreatorPage from "./pages/ChatbotAndImageCreatorPages/ImageCreatorPage";
import HomePage from "./pages/HomePage/HomePage";
import NotFoundPage from "./pages/NotFoundPage/NotFoundPage";
import CartPage from "./pages/UserPages/CartPage";
import CheckoutPage from "./pages/UserPages/CheckoutPage";
import MyAccountPage from "./pages/UserPages/MyAccountPage";
import MyWishlistPage from "./pages/UserPages/MyWishlistPage";
import NotificationsPage from "./pages/UserPages/NotificationsPage";
import CouponsPage from "./pages/UserPages/CouponsPage";
import MyOrdersPage from "./pages/UserPages/MyOrdersPage";
import LandingPage from "./pages/LoginAndSignUpPages/LandingPage";
import LoginPage from "./pages/LoginAndSignUpPages/LoginPage";
import SignUpPage from "./pages/LoginAndSignUpPages/SignUpPage";

import AppNavBar from "./components/UtilsComponents/AppNavBar";
import SideBar from "./components/UtilsComponents/SideBar";
import SizeRecPopup from "./components/UtilsComponents/SizeRecPopup";

import "./styles/appStyles/App.css";

export default function App() {
  /**
   * User State
   */
  const [isUserSignedIn, setIsUserSignedIn] = useState(false);

  /**
   * Size Recommendation Popup States
   */
  const [isSizePopupOpened, setIsSizePopupOpened] = useState(false);

  /**
   * SideBar States & Functions
   */
  const [isAppDrawerOpened, setIsAppDrawerOpened] = useState(false);
  const handleDrawerIconClick = () => {
    if (isAppDrawerOpened) {
      setIsAppDrawerOpened(false);
      return;
    }
    setIsAppDrawerOpened(true);
  };

  /**
   * Cart States
   */
  const [cartItemsCount, setCartItemsCount] = useState(0);

  /**
   * Categories States & Functions
   */
  const [categories, setCategories] = useState([]);
  useEffect(() => {
    // Categories
    const fetchCategories = async () => {
      const data = await getAllCategories();
      setCategories(data);
    };

    fetchCategories();
  }, []);

  /**
   * Unread Notifications
   */
  const [unreadNotifications, setUnreadNotifications] = useState(0);
  const [isNotificationsPageLoading, setIsNotificationsPageLoading] =
    useState(true);
  useEffect(() => {
    // Notifications
    const fetchNotifications = async () => {
      try {
        const data = await getUnreadNotifications();
        setUnreadNotifications(data.reverse());
      } catch (error) {
        console.log(`Error while fetching the notifications`);
      } finally {
        setIsNotificationsPageLoading(false);
      }
    };

    fetchNotifications();
  }, []);

  return (
    <>
      <BrowserRouter>
        {/* Utils Components */}
        {isUserSignedIn && (
          <AppNavBar
            cartItemsCount={cartItemsCount}
            setIsSizePopupOpened={setIsSizePopupOpened}
            isAppDrawerOpened={isAppDrawerOpened}
            handleDrawerIconClick={handleDrawerIconClick}
            unreadNotifications={unreadNotifications}
            setIsUserSignedIn={setIsUserSignedIn}
          />
        )}
        {isUserSignedIn && (
          <SideBar
            isAppDrawerOpened={isAppDrawerOpened}
            setIsAppDrawerOpened={setIsAppDrawerOpened}
            handleDrawerIconClick={handleDrawerIconClick}
            setIsSizePopupOpened={setIsSizePopupOpened}
          />
        )}
        {isSizePopupOpened && (
          <SizeRecPopup setIsSizePopupOpened={setIsSizePopupOpened} />
        )}

        {/* Landing, Login, and Sign up Pages */}
        <Routes>
          {!isUserSignedIn ? (
            // Landing Page
            <Route path="/" element={<LandingPage />} />
          ) : (
            // Home Page
            <Route
              path="/"
              element={<HomePage setCartItemsCount={setCartItemsCount} />}
            />
          )}
          <Route
            path="/login"
            element={<LoginPage setIsUserSignedIn={setIsUserSignedIn} />}
          />
          <Route
            path="/sign-up"
            element={<SignUpPage setIsUserSignedIn={setIsUserSignedIn} />}
          />

          {/* Chatbot & Image Creator Pages */}
          <Route
            path="/ChatbotAIAndImgCreatorPage"
            element={<ChatbotAIAndImgCreatorPage />}
          />
          <Route path="/chatbot" element={<ChatbotPage />} />
          <Route path="/imageCreator" element={<ImageCreatorPage />} />

          {/* Products Page */}
          <Route
            path="/products/rec-items"
            element={<ProductsPage pageTitle={"Recommended Items"} />}
          />

          <Route
            path="/products/best-sellers"
            element={<ProductsPage pageTitle={"Best Sellers"} />}
          />

          {categories.map((category, index) => {
            return (
              <Route
                key={index}
                path={`/categories/${category.title}`}
                element={
                  <ProductsPage
                    pageTitle={category.title}
                    image={category.image}
                  />
                }
              />
            );
          })}

          <Route
            path="/products/product-details"
            element={<ProductDetailsPage />}
          />

          {/* Categories Page */}
          <Route path="/categories" element={<CategoriesPage />} />

          {/* Cart Pages */}
          <Route path="user/cart" element={<CartPage />} />
          <Route path="user/cart/checkout" element={<CheckoutPage />} />

          {/* User Pages */}
          <Route path="user/account" element={<MyAccountPage />} />
          <Route
            path="user/notifications"
            element={
              <NotificationsPage
                unreadNotifications={unreadNotifications}
                isNotificationsPageLoading={isNotificationsPageLoading}
              />
            }
          />
          <Route path="user/wishlist" element={<MyWishlistPage />} />
          <Route path="user/coupons" element={<CouponsPage />} />
          <Route path="user/my-orders" element={<MyOrdersPage />} />

          {/* Page Not Found */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}
