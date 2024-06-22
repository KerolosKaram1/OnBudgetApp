// Validate name
const isNameValid = (name) => {
  // Add your validation logic for the name
  return name.trim() !== "";
};

// Validate phone
const isPhoneValid = (phone) => {
  // Add your validation logic for the phone
  return phone.trim() !== "";
};

// Validate email
const isEmailValid = (email) => {
  // Add your validation logic for the email
  return email.trim() !== "";
};

// Validate password
const isPasswordValid = (password) => {
  // Add your validation logic for the password
  return password.trim() !== "";
};

// Validate confirm password
const isConfirmPasswordValid = (password, confirmPassword) => {
  // Add your validation logic for confirming password
  return password === confirmPassword;
};

// Validate user type
const isUserTypeValid = (userType) => {
  // Add your validation logic for the user type
  return userType !== "";
};

// Validate gender
const isGenderValid = (gender) => {
  // Add your validation logic for the gender
  return gender !== "";
};

let b;
export const isFormValid = (
  userGender,
  name,
  email,
  phone,
  password,
  confirmPassword,
  userType,
  policyCheckBox
) => {
  b =
    isNameValid(name) &&
    isPhoneValid(phone) &&
    isEmailValid(email) &&
    isPasswordValid(password) &&
    isConfirmPasswordValid(password, confirmPassword) &&
    isUserTypeValid(userType) &&
    isGenderValid(userGender) &&
    policyCheckBox;

  return b;
};
