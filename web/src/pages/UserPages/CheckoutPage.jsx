import { useState } from "react";
import {
  Container,
  Typography,
  Box,
  FormControl,
  RadioGroup,
  FormControlLabel,
  Radio,
  TextField,
  Button,
  Divider,
} from "@mui/material";

export default function CheckoutPage({ totalOrderPrice }) {
  const [paymentMethod, setPaymentMethod] = useState("cash");
  const [cardNumber, setCardNumber] = useState("");
  const [cardName, setCardName] = useState("");
  const [expiryDate, setExpiryDate] = useState("");
  const [cvv, setCvv] = useState("");
  const [addressOption, setAddressOption] = useState("profile");
  const [newAddress, setNewAddress] = useState("");

  const handlePaymentMethodChange = (event) => {
    setPaymentMethod(event.target.value);
  };

  const handleCardNumberChange = (event) => {
    setCardNumber(event.target.value);
  };

  const handleCardNameChange = (event) => {
    setCardName(event.target.value);
  };

  const handleExpiryDateChange = (event) => {
    setExpiryDate(event.target.value);
  };

  const handleCvvChange = (event) => {
    setCvv(event.target.value);
  };

  const handleAddressOptionChange = (event) => {
    setAddressOption(event.target.value);
  };

  const handleNewAddressChange = (event) => {
    setNewAddress(event.target.value);
  };

  const handleSubmit = () => {
    // Handle form submission logic here
    alert("Order placed successfully!");
  };

  return (
    <main style={{ marginTop: "5rem", marginLeft: "5rem" }}>
      <Container maxWidth="sm" sx={{ paddingBottom: "1rem" }}>
        <Typography variant="h4" gutterBottom>
          Checkout
        </Typography>

        <Divider sx={{ marginBottom: "2rem" }} />

        {/* Select Address Section */}
        <FormControl component="fieldset" fullWidth>
          <Typography variant="h6">Select Address:</Typography>
          <RadioGroup
            aria-label="address-option"
            name="address-option"
            value={addressOption}
            onChange={handleAddressOptionChange}
          >
            <FormControlLabel
              value="profile"
              control={<Radio />}
              label="Use Address from Profile"
            />
            <FormControlLabel
              value="new"
              control={<Radio />}
              label="Add a New Address"
            />
          </RadioGroup>
        </FormControl>

        {addressOption === "new" && (
          <Box sx={{ marginTop: "2rem" }}>
            <Typography variant="h6" gutterBottom>
              New Address
            </Typography>
            <TextField
              label="Enter New Address"
              fullWidth
              value={newAddress}
              onChange={handleNewAddressChange}
              margin="normal"
            />
          </Box>
        )}

        <Divider
          sx={{
            marginBottom: "2rem",
            marginTop: "2rem",
            borderBottomWidth: "3px",
            borderColor: "primary.main",
          }}
        />

        {/* Payment Method Section */}
        <FormControl component="fieldset" fullWidth>
          <Typography variant="h6">Select Payment Method:</Typography>
          <RadioGroup
            aria-label="payment-method"
            name="payment-method"
            value={paymentMethod}
            onChange={handlePaymentMethodChange}
          >
            <FormControlLabel
              value="cash"
              control={<Radio />}
              label="Cash on Delivery"
            />
            <FormControlLabel value="visa" control={<Radio />} label="Visa" />
          </RadioGroup>
        </FormControl>

        {paymentMethod === "visa" && (
          <Box sx={{ marginTop: "2rem" }}>
            <Typography variant="h6" gutterBottom>
              Visa Information
            </Typography>
            <TextField
              label="Card Number"
              fullWidth
              value={cardNumber}
              onChange={handleCardNumberChange}
              margin="normal"
            />
            <TextField
              label="Card Name"
              fullWidth
              value={cardName}
              onChange={handleCardNameChange}
              margin="normal"
            />
            <TextField
              label="Expiry Date (MM/YY)"
              fullWidth
              value={expiryDate}
              onChange={handleExpiryDateChange}
              margin="normal"
            />
            <TextField
              label="CVV"
              fullWidth
              value={cvv}
              onChange={handleCvvChange}
              margin="normal"
            />
          </Box>
        )}

        <Box sx={{ marginTop: "2rem" }}>
          <Typography variant="h6">
            {/* Total Order Price: ${totalOrderPrice.toFixed(2)} */}
            Total Order Price: $350.50
          </Typography>
        </Box>

        <Box sx={{ marginTop: "2rem" }}>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleSubmit}
          >
            Place Order
          </Button>
        </Box>
      </Container>
    </main>
  );
}
