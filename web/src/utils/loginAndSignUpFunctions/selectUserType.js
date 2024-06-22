export function selectCustomer(customer, seller, setUserType) {
  if (customer.current.style.border === "4px solid rgb(255, 205, 78)") {
    customer.current.style.border = "none";
    setUserType("");
  } else {
    customer.current.style.border = "4px solid rgb(255, 205, 78)";
    seller.current.style.border = "none";
    setUserType("customer");
  }
}

export function selectSeller(customer, seller, setUserType) {
  if (seller.current.style.border === "4px solid rgb(255, 205, 78)") {
    seller.current.style.border = "none";
    setUserType("");
  } else {
    seller.current.style.border = "4px solid rgb(255, 205, 78)";
    customer.current.style.border = "none";
    setUserType("seller");
  }
}
