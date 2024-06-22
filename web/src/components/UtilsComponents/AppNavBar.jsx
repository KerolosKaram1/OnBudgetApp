import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { getCartItemsCount } from "../../utils/getCartItems";

import {
  styled,
  alpha,
  ThemeProvider,
  createTheme,
} from "@mui/material/styles";

import {
  AppBar,
  Avatar,
  Badge,
  Box,
  IconButton,
  InputBase,
  Menu,
  MenuItem,
  Toolbar,
  Tooltip,
  Typography,
} from "@mui/material";
import { tooltipClasses } from "@mui/material/Tooltip";

import MenuIcon from "@mui/icons-material/Menu";
import SearchIcon from "@mui/icons-material/Search";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import NotificationsIcon from "@mui/icons-material/Notifications";

const theme = createTheme({
  palette: {
    primary: {
      main: "#006880",
    },
  },
});

const Search = styled("div")(({ theme }) => ({
  position: "relative",
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.common.white, 0.15),
  "&:hover": {
    backgroundColor: alpha(theme.palette.common.white, 0.25),
  },
  marginRight: theme.spacing(2),
  marginLeft: 0,
  width: "100%",
  [theme.breakpoints.up("sm")]: {
    marginLeft: theme.spacing(3),
    width: "auto",
  },
}));

const SearchIconWrapper = styled("div")(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: "100%",
  position: "absolute",
  pointerEvents: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: "inherit",
  "& .MuiInputBase-input": {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create("width"),
    width: "100%",
    [theme.breakpoints.up("md")]: {
      width: "50ch",
    },
  },
}));

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

export default function AppNavBar({
  setIsSizePopupOpened,
  isAppDrawerOpened,
  handleDrawerIconClick,
  unreadNotifications,
  setIsUserSignedIn,
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [mobileMoreAnchorEl, setMobileMoreAnchorEl] = useState(null);
  const [notificationsAnchorEl, setNotificationsAnchorEl] = useState(null);
  const [cartItemsCount, setCartItemsCount] = useState(0);

  const [showAppBar, setShowAppBar] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);
  const navigate = useNavigate();

  const isMenuOpen = Boolean(anchorEl);
  const isMobileMenuOpen = Boolean(mobileMoreAnchorEl);
  const isNotificationsMenuOpen = Boolean(notificationsAnchorEl);

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleNotificationsMenuOpen = (event) => {
    setNotificationsAnchorEl(event.currentTarget);
  };

  const handleMobileMenuClose = () => {
    setMobileMoreAnchorEl(null);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    handleMobileMenuClose();
  };

  const handleNotificationsMenuClose = () => {
    setNotificationsAnchorEl(null);
  };

  const handleMobileMenuOpen = (event) => {
    setMobileMoreAnchorEl(event.currentTarget);
  };

  const handleScroll = () => {
    const currentScrollY = window.scrollY;
    if (currentScrollY > lastScrollY && currentScrollY > 50) {
      setShowAppBar(false);
    } else {
      setShowAppBar(true);
    }
    setLastScrollY(currentScrollY);
  };

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastScrollY]);

  // const handleDeleteNotification = (id) => {
  //   setNotifications(
  //     notifications.filter((notification) => notification.notificationID !== id)
  //   );
  // };

  useEffect(() => {
    // Cart Items
    const fetchCartItemsCount = async () => {
      const data = await getCartItemsCount();
      setCartItemsCount(data);
    };

    Promise.all([fetchCartItemsCount()]).catch((error) => {
      console.error("Error fetching products:", error);
    });
  }, []);

  const menuId = "more-options-menu";
  const renderMenu = (
    <Menu
      anchorEl={anchorEl}
      anchorOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      id={menuId}
      keepMounted
      transformOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      open={isMenuOpen}
      onClose={handleMenuClose}
    >
      <Avatar
        aria-label="account of current user"
        src={"resources/images/profile7.jpg"}
        alt={"Profile Image"}
        sx={{
          width: 60,
          height: 60,
          marginTop: 0.6,
          marginBottom: 1,
          marginLeft: "50px",
          textAlign: "center",
        }}
      />
      <Typography
        variant="h6"
        noWrap
        style={{
          color: "black",
          textAlign: "center",
          marginInline: "15px",
          fontWeight: "bold",
        }}
      >
        Mario Sarhan
      </Typography>
      <Typography
        noWrap
        style={{
          color: "black",
          textAlign: "center",
          // fontWeight: "bold",
          marginInline: "15px",
        }}
      >
        Balance: $260.50
      </Typography>
      <hr />
      <MenuItem
        onClick={() => {
          handleMenuClose();
          navigate("/user/account");
        }}
      >
        My Account
      </MenuItem>
      <MenuItem
        onClick={() => {
          handleMenuClose();
          navigate("/user/coupons");
        }}
      >
        Coupons
      </MenuItem>
      <MenuItem
        onClick={() => {
          handleMenuClose();
          navigate("/user/my-orders");
        }}
      >
        My Orders
      </MenuItem>
      <MenuItem
        onClick={() => {
          handleMenuClose();
          setIsUserSignedIn(false);
        }}
      >
        Logout
      </MenuItem>
    </Menu>
  );

  const notificationsMenuId = "notifications-menu";
  const renderNotificationsMenu = (
    <Menu
      anchorEl={notificationsAnchorEl}
      anchorOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      id={notificationsMenuId}
      keepMounted
      transformOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      open={isNotificationsMenuOpen}
      onClose={handleNotificationsMenuClose}
      sx={{ width: "50%" }}
    >
      <Typography
        variant="h6"
        style={{
          textAlign: "center",
          fontWeight: "bold",
          marginInline: "15px",
        }}
      >
        Notifications
      </Typography>

      <hr />

      {unreadNotifications.length > 0 ? (
        unreadNotifications.map((notification) => (
          <MenuItem
            key={notification.id}
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: { xs: "8px 8px", sm: "8px 16px" }, // Adjust padding based on screen size
              width: "100%", // Ensure it takes the full width of its container
            }}
            onClick={() => {
              handleMobileMenuClose();
              handleNotificationsMenuClose();
              navigate("/user/notifications");
            }}
          >
            <Typography
              variant="body2"
              sx={{
                flexGrow: 1,
                marginRight: 2,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap", // Ensure text remains on a single line
                fontSize: { xs: "0.8rem", sm: "0.875rem" }, // Adjust font size based on screen size
              }}
            >
              {notification.notificationText}
            </Typography>
          </MenuItem>
        ))
      ) : (
        <Typography variant="body2" style={{ textAlign: "center" }}>
          No new notifications
        </Typography>
      )}
    </Menu>
  );

  const mobileMenuId = "more-options-menu-mobile";
  const renderMobileMenu = (
    <Menu
      anchorEl={mobileMoreAnchorEl}
      anchorOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      id={mobileMenuId}
      keepMounted
      transformOrigin={{
        vertical: "top",
        horizontal: "right",
      }}
      open={isMobileMenuOpen}
      onClose={handleMobileMenuClose}
    >
      <MenuItem
        onClick={() => {
          handleMobileMenuClose();
          navigate("/user/cart");
        }}
      >
        <IconButton size="large" aria-label="go to cart page" color="inherit">
          <Badge badgeContent={cartItemsCount} color="error">
            <ShoppingCartIcon />
          </Badge>
        </IconButton>
        <p>Shopping Cart</p>
      </MenuItem>
      <MenuItem onClick={handleNotificationsMenuOpen}>
        <IconButton
          size="large"
          aria-label="show new notifications"
          color="inherit"
        >
          <Badge badgeContent={unreadNotifications.length} color="error">
            <NotificationsIcon />
          </Badge>
        </IconButton>
        <p>Notifications</p>
      </MenuItem>
      <MenuItem onClick={handleProfileMenuOpen}>
        <Avatar
          aria-label="account of current user"
          src={"resources/images/profile7.jpg"}
          alt={"profile image"}
          sx={{
            width: 40,
            height: 40,
            marginRight: 1.2,
            cursor: "pointer",
          }}
        />
        <p>More Options</p>
      </MenuItem>
    </Menu>
  );

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ flexGrow: 1 }}>
        <AppBar
          position="fixed"
          sx={{
            transition: "top 0.3s",
            top: showAppBar ? "0" : "-64px", // Adjust the value -64px based on the height of your AppBar
            // zIndex: open ? 1200 : 1300, // Conditionally set z-index
          }}
        >
          <Toolbar>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="open drawer"
              sx={{ mr: 2, ...(isAppDrawerOpened && { display: "none" }) }}
              onClick={handleDrawerIconClick}
            >
              <MenuIcon />
            </IconButton>
            <Typography
              onClick={() => navigate("/")}
              variant="h6"
              noWrap
              component="div"
              sx={{
                display: { xs: "none", sm: "block" },
                userSelect: "none",
                fontWeight: "bold",
                fontStyle: "italic",
                fontFamily: "comic-sans-serif",
                fontSize: "24px",
                cursor: "pointer",
              }}
            >
              OnBudget
            </Typography>
            <Search>
              <SearchIconWrapper>
                <SearchIcon />
              </SearchIconWrapper>
              <StyledInputBase
                placeholder="Search…"
                inputProps={{ "aria-label": "search" }}
              />
            </Search>
            <Box sx={{ flexGrow: 1 }} />
            <Box sx={{ display: { xs: "none", md: "flex" } }}>
              <CustomTooltip title={"Shopping Cart"} placement="bottom">
                <IconButton
                  size="large"
                  aria-label={`shopping cart`}
                  color="inherit"
                  onClick={() => navigate("/user/cart")}
                >
                  <Badge badgeContent={cartItemsCount} color="error">
                    <ShoppingCartIcon />
                  </Badge>
                </IconButton>
              </CustomTooltip>
              <CustomTooltip title={"Show Notifications"} placement="bottom">
                <IconButton
                  size="large"
                  aria-label={`show new notifications`}
                  color="inherit"
                  onClick={handleNotificationsMenuOpen}
                >
                  <Badge
                    badgeContent={unreadNotifications.length}
                    color="error"
                  >
                    <NotificationsIcon />
                  </Badge>
                </IconButton>
              </CustomTooltip>
              <CustomTooltip title={"see more"} placement="bottom">
                <Avatar
                  aria-label="account of current user"
                  src={"resources/images/profile7.jpg"}
                  alt={"profile image"}
                  sx={{
                    width: 40,
                    height: 40,
                    marginLeft: 1,
                    cursor: "pointer",
                  }}
                  onClick={handleProfileMenuOpen}
                />
              </CustomTooltip>
            </Box>
            <Box sx={{ display: { xs: "flex", md: "none" } }}>
              <IconButton
                size="large"
                aria-label="show more"
                aria-controls={mobileMenuId}
                aria-haspopup="true"
                onClick={handleMobileMenuOpen}
                color="inherit"
              >
                <Avatar
                  aria-label="account of current user"
                  src={"resources/images/profile7.jpg"}
                  alt={"profile image"}
                  sx={{
                    width: 40,
                    height: 40,
                    marginLeft: 1,
                    cursor: "pointer",
                  }}
                />
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>
        {renderMobileMenu}
        {renderMenu}
        {renderNotificationsMenu}
      </Box>
    </ThemeProvider>
  );
}
