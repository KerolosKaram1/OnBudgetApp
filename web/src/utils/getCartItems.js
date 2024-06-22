export const getCartItems = async () => {
  try {
    const response = await fetch("https://fakestoreapi.com/products", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    // User Error
    if (!response) {
      return "We cannot find any products";
    }

    // Success
    return await response.json();
  } catch (error) {
    // Server Error
    return `There was a problem with the server connection: ${error.message}`;
  }

  // return [];
};

export const getCartItemsCount = async () => {
  const res = await getCartItems();
  return res.length;
};