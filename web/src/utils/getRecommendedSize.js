export default async function getRecommendedSize(age, height, weight) {
  const APIRequestBody = {
    weight: weight.trim(),
    age: age.trim(),
    height: height.trim(),
  };

  try {
    const response = await fetch("http://127.0.0.1:5000/submit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(APIRequestBody),
    });

    // User Error
    if (!response) {
      return "Invalid Input";
    }

    // Success
    return await response.json();
  } catch (error) {
    // Server Error
    return `There was a problem with the connection: ${error.message}`;
  }
}
