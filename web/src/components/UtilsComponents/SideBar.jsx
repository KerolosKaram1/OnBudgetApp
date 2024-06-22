import * as React from "react";
import { useNavigate } from "react-router-dom";

import { styled } from "@mui/material/styles";

import Typography from "@mui/material/Typography";
import MuiDrawer from "@mui/material/Drawer";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import HomeIcon from "@mui/icons-material/Home";
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart";
import FavoriteIcon from "@mui/icons-material/Favorite";
import HeightIcon from "@mui/icons-material/Height";
import AssistantIcon from "@mui/icons-material/Assistant";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import CloseIcon from "@mui/icons-material/Close";
import MenuIcon from "@mui/icons-material/Menu";

const drawerWidth = 240;

const openedMixin = (theme) => ({
  width: drawerWidth,
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: "hidden",
});

const closedMixin = (theme) => ({
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: "hidden",
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up("sm")]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-end",
  padding: theme.spacing(0, 1),
  ...theme.mixins.toolbar,
}));

const Drawer = styled(MuiDrawer, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: "nowrap",
  boxSizing: "border-box",
  zIndex: theme.zIndex.drawer + 100, // Ensure drawer is above AppBar
  ...(open && {
    ...openedMixin(theme),
    "& .MuiDrawer-paper": openedMixin(theme),
    // zIndex: theme.zIndex.drawer + 2, // Ensure drawer paper is above AppBar
  }),
  ...(!open && {
    ...closedMixin(theme),
    "& .MuiDrawer-paper": closedMixin(theme),
  }),
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

function DrawerItemsIcons({ text }) {
  if (text === "Home") {
    return <HomeIcon />;
  } else if (text === "Cart") {
    return <ShoppingCartIcon />;
  } else if (text === "Wishlist") {
    return <FavoriteIcon />;
  } else if (text === "Cart") {
    return <ShoppingCartIcon />;
  } else if (text === "My Account") {
    return <AccountCircleIcon />;
  } else if (text === "Chatbot AI") {
    return <AssistantIcon />;
  } else if (text === "Size Recommendation") {
    return <HeightIcon />;
  }
}

export default function SideBar({
  isAppDrawerOpened,
  setIsAppDrawerOpened,
  handleDrawerIconClick,
  setIsSizePopupOpened,
}) {
  const navigate = useNavigate();

  return (
    <Drawer variant="permanent" open={isAppDrawerOpened}>
      <DrawerHeader>
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{
            mr: 4,
            display: { sm: "block" },
            userSelect: "none",
            fontWeight: "bold",
            fontStyle: "italic",
            fontFamily: "comic-sans-serief",
            fontSize: "24px",
            cursor: "pointer",
          }}
          onClick={() => navigate("/")}
        >
          OnBudget
        </Typography>
        <IconButton onClick={handleDrawerIconClick}>
          {isAppDrawerOpened ? <CloseIcon /> : <MenuIcon />}
        </IconButton>
      </DrawerHeader>
      <Divider />
      <List>
        {["Home", "Cart", "Wishlist"].map((text, index) => (
          <ListItem key={text} disablePadding sx={{ display: "block" }}>
            <CustomTooltip title={text} placement="right">
              <ListItemButton
                sx={{
                  // TODO: backgroundColor: text === "Home" ? "gcrey" : "white",
                  minHeight: 48,
                  justifyContent: isAppDrawerOpened ? "initial" : "center",
                  px: 2.5,
                }}
                onClick={() => {
                  if (text === "Home") {
                    navigate("/");
                  } else if (text === "Cart") {
                    navigate("/user/cart");
                  } else if (text === "Wishlist") {
                    navigate("/user/wishlist");
                  }
                  setIsAppDrawerOpened(false);
                }}
              >
                <ListItemIcon
                  sx={{
                    minWidth: 0,
                    mr: isAppDrawerOpened ? 3 : "auto",
                    justifyContent: "center",
                  }}
                >
                  <DrawerItemsIcons text={text} />
                </ListItemIcon>
                <ListItemText
                  primary={text}
                  sx={{ opacity: isAppDrawerOpened ? 1 : 0 }}
                />
              </ListItemButton>
            </CustomTooltip>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        {["My Account", "Chatbot AI", "Size Recommendation"].map(
          (text, index) => (
            <ListItem key={text} disablePadding sx={{ display: "block" }}>
              <CustomTooltip title={text} placement="right">
                <ListItemButton
                  sx={{
                    // TODO: backgroundColor: text === "Home" ? "grey" : "white",
                    minHeight: 48,
                    justifyContent: isAppDrawerOpened ? "initial" : "center",
                    px: 2.5,
                  }}
                  onClick={() => {
                    if (text === "My Account") {
                      navigate("/user/account");
                    } else if (text === "Chatbot AI") {
                      navigate("/ChatbotAIAndImgCreatorPage");
                    } else if (text === "Size Recommendation") {
                      setIsSizePopupOpened(true);
                    }
                    setIsAppDrawerOpened(false);
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: isAppDrawerOpened ? 3 : "auto",
                      justifyContent: "center",
                    }}
                  >
                    <DrawerItemsIcons text={text} />
                  </ListItemIcon>
                  <ListItemText
                    primary={text}
                    sx={{ opacity: isAppDrawerOpened ? 1 : 0 }}
                  />
                </ListItemButton>
              </CustomTooltip>
            </ListItem>
          )
        )}
      </List>
    </Drawer>
  );
}
