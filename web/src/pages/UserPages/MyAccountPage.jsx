import React, { useState, useEffect } from "react";
import {
  Container,
  Typography,
  Divider,
  TextField,
  Button,
  Box,
  IconButton,
  InputAdornment,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Input,
  DialogContentText,
} from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";

const initialUserData = {
  userImg: "/resources/images/profile7.jpg",
  fullName: "John Doe",
  email: "john.doe@example.com",
  currentPassword: "password123",
  phone: "123-456-7890",
  address: "123 Main St, Anytown, USA",
};

export default function MyAccountPage() {
  const [userData, setUserData] = useState(initialUserData);
  const [isSaveEnabled, setIsSaveEnabled] = useState(false);
  const [isEdited, setIsEdited] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [avatarUrl, setAvatarUrl] = useState(initialUserData.userImg); // State for avatar image URL
  const [openDialog, setOpenDialog] = useState(false); // State for dialog visibility
  const [newAvatarUrl, setNewAvatarUrl] = useState(""); // State for input field value

  useEffect(() => {
    const isChanged =
      userData.userImg !== initialUserData.userImg ||
      userData.fullName !== initialUserData.fullName ||
      userData.email !== initialUserData.email ||
      userData.currentPassword !== initialUserData.currentPassword ||
      userData.phone !== initialUserData.phone ||
      userData.address !== initialUserData.address;

    setIsSaveEnabled(isChanged);
    setIsEdited(isChanged);
  }, [userData]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setUserData((prevUserData) => ({
      ...prevUserData,
      [name]: value,
    }));
  };

  const handleSave = () => {
    // Save changes
    console.log("Saved changes:", userData);
    // Optionally update initialUserData if needed
    // setInitialUserData(userData);
  };

  const handleClickShowPassword = () => {
    setShowPassword(!showPassword);
  };

  const handleDiscardChanges = () => {
    setUserData(initialUserData);
    setIsEdited(false);
  };

  const handleAvatarClick = () => {
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    // Reset input field value
    setNewAvatarUrl("");
  };

  const handleAvatarChange = () => {
    setAvatarUrl(newAvatarUrl);
    handleCloseDialog();
    // Optionally save the avatar URL to user data or elsewhere
    // console.log("New avatar URL:", newAvatarUrl);
  };

  return (
    <main style={{ marginTop: "5rem", marginLeft: "5rem" }}>
      <Container maxWidth="lg" sx={{ paddingBottom: "1rem" }}>
        <Typography variant="h4" gutterBottom>
          My Account
        </Typography>

        <Divider sx={{ marginBottom: "2rem" }} />

        <Box component="form" sx={{ display: "grid", gap: 2 }}>
          <Avatar
            alt="User Avatar"
            src={avatarUrl}
            sx={{ width: 100, height: 100, cursor: "pointer", marginBottom: 2 }}
            onClick={handleAvatarClick}
          />

          <TextField
            label="Full Name"
            variant="outlined"
            name="fullName"
            value={userData.fullName}
            onChange={handleChange}
            fullWidth
          />
          <TextField
            label="Email"
            variant="outlined"
            name="email"
            value={userData.email}
            onChange={handleChange}
            fullWidth
          />
          <TextField
            label="Current Password"
            variant="outlined"
            type={showPassword ? "text" : "password"}
            name="currentPassword"
            value={userData.currentPassword}
            onChange={handleChange}
            fullWidth
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    aria-label="toggle password visibility"
                    onClick={handleClickShowPassword}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <TextField
            label="Phone"
            variant="outlined"
            name="phone"
            value={userData.phone}
            onChange={handleChange}
            fullWidth
          />
          <TextField
            label="Address"
            variant="outlined"
            name="address"
            value={userData.address}
            onChange={handleChange}
            fullWidth
          />

          <Button
            variant="contained"
            color="primary"
            onClick={handleSave}
            disabled={!isSaveEnabled}
            sx={{ marginTop: "1rem" }}
          >
            Save
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleDiscardChanges}
            disabled={!isEdited}
            sx={{ marginTop: "1rem" }}
          >
            Discard Changes
          </Button>
        </Box>

        <Dialog open={openDialog} onClose={handleCloseDialog}>
          <DialogTitle>Update Avatar</DialogTitle>
          <DialogContent>
            <DialogContentText style={{ marginBottom: "1rem" }}>
              Please enter the URL for the new avatar image:
            </DialogContentText>
            <FormControl fullWidth>
              <InputLabel htmlFor="avatar-url">Avatar URL</InputLabel>
              <Input
                id="avatar-url"
                type="text"
                value={newAvatarUrl}
                onChange={(e) => setNewAvatarUrl(e.target.value)}
              />
            </FormControl>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog} color="primary">
              Cancel
            </Button>
            <Button onClick={handleAvatarChange} color="primary">
              Update
            </Button>
          </DialogActions>
        </Dialog>
      </Container>
    </main>
  );
}
