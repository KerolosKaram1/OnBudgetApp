import { Typography, Paper, Grid } from "@mui/material";

export default function CartItem({ notification }) {
  return (
    <>
      <Paper elevation={3} style={{ padding: "16px", marginBottom: "16px" }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={5}>
            {notification.notificationText && (
              <Typography
                variant="h6"
                sx={{
                  marginBottom: 1,
                  fontWeight: "bold",
                }}
              >
                {notification.notificationText}
              </Typography>
            )}

            {notification.notificationDate && (
              <Typography variant="h6" style={{ marginBottom: 10 }}>
                Date: {notification.notificationDate}
              </Typography>
            )}
          </Grid>
        </Grid>
      </Paper>
    </>
  );
}
