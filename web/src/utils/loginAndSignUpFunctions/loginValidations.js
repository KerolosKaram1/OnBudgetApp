const validateEmailOrPhone = (emailValue) => {
  // Add your validation logic
  return emailValue && emailValue.trim() !== "";
};
const validatePassword = (passValue) => {
  // Add your validation logic
  return passValue && passValue.trim() !== "";
};

export const isFormValid = (emailValue, passValue) => {
  return validateEmailOrPhone(emailValue) && validatePassword(passValue);
};
