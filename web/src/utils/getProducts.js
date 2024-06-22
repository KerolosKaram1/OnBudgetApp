// eslint-disable-next-line no-unused-vars
const BASE_URL = process.env.REACT_APP_BASE_URL;

export const getRecommendedProducts = async () => {
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
};

export const getBestSellersProducts = async () => {
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
};

export const getCartProducts = async () => {
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
};

export const getWishlistProducts = async () => {
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
};

export const getProductDetailsByID = async (id) => {
  try {
    const response = await fetch(`https://fakestoreapi.com/products/${id}`, {
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
};

export const getAllProductsInThisCategory = async (category) => {
  // try {
  //   const response = await fetch(`${BASE_URL}/Category/${category}`, {
  //     method: "GET",
  //     headers: {
  //       "Content-Type": "application/json",
  //     },
  //   });

  //   // User Error
  //   if (!response) {
  //     return "Sorry, There are no products in this category 😥";
  //   }

  //   // Success
  //   return await response.json();
  // } catch (error) {
  //   // Server Error
  //   return `There was a problem with the server connection: ${error.message}`;
  // }

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
};

/**
 * API response
 
 {
    "id": 1,
    "title": "Fjallraven - Foldsack No. 1 Backpack, Fits 15 Laptops",
    "price": 109.95,
    "description": "Your perfect pack for everyday use and walks in the forest. Stash your laptop (up to 15 inches) in the padded sleeve, your everyday",
    "category": "men's clothing",
    "image": "https://fakestoreapi.com/img/81fPKd-2AYL._AC_SL1500_.jpg",
    "rating": {
      "rate": 3.9,
      "count": 120
    }
  }

 */
