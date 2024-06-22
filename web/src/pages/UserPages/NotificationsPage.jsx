import Notification from "./../../components/NotificationsComponents/Notification";

import {
  Container,
  Typography,
  Box,
  CircularProgress,
  Divider,
} from "@mui/material";

export default function NotificationsPage({
  unreadNotifications,
  isNotificationsPageLoading,
}) {
  return (
    <main
      style={{ marginTop: "5rem", marginLeft: "5rem", marginBottom: "5rem" }}
    >
      <Container maxWidth="lg">
        {/* Page Title */}
        <Typography variant="h4" gutterBottom>
          My Notifications 😃
        </Typography>

        <Divider sx={{ marginBottom: "2rem" }} />

        {/* Display loading indicator if fetching data */}
        {isNotificationsPageLoading ? (
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
        ) : (
          <>
            <Box marginTop={2}>
              {/* Cart items if they're exist */}
              {unreadNotifications.length > 0 ? (
                unreadNotifications.map((notification) => (
                  <Notification
                    key={notification.notificationID}
                    notification={notification}
                  />
                ))
              ) : (
                // Display this message if there're no cart items
                <Typography variant="h5">You Have No Notifications!</Typography>
              )}
            </Box>
          </>
        )}
      </Container>
    </main>
  );
}
