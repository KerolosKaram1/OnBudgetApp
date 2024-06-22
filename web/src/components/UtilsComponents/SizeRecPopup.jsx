import { useRef, useState, useEffect } from "react";
import "../../styles/utilsStyles/SizeRecPopup.css";
import getRecommendedSize from "../../utils/getRecommendedSize";

export default function Popup({ setIsSizePopupOpened }) {
  const ageInput = useRef(null);
  const heightInput = useRef(null);
  const weightInput = useRef(null);
  const errorLabel = useRef(null);

  const [age, setAge] = useState("");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [APIRequestBody, setAPIRequestBody] = useState([]);
  const [APIResponseMsg, setAPIResponseMsg] = useState("");
  const [inputPopupOrResultPopup, setInputPopupOrResultPopup] =
    useState("input");

  // Input fields change
  const handleAgeChange = (event) => {
    setAge(event.target.value);
  };
  const handleHeightChange = (event) => {
    setHeight(event.target.value);
  };
  const handleWeightChange = (event) => {
    setWeight(event.target.value);
  };

  // Submit
  function handleFormSubmission(e) {
    e.preventDefault();

    // input fields validations
    const regex = /^[0-9]+$/;
    if (age.trim() === "" || height.trim() === "" || weight.trim() === "") {
      errorLabel.current.innerText = "Please, fill in all the data!";
      return;
    } else if (
      !regex.test(age.trim()) ||
      !regex.test(height.trim()) ||
      !regex.test(weight.trim())
    ) {
      errorLabel.current.innerText = "Please, enter a valid data!";
      return;
    }

    // close the input popup and open the result popup
    setInputPopupOrResultPopup("result");
    setAPIRequestBody([age, height, weight]);
  }

  // Cancel
  function handleFormCancelation(e) {
    e.preventDefault();
    setIsSizePopupOpened(false);
  }

  // Enter data again
  function handleEnterAgainBtn(e) {
    e.preventDefault();
    setInputPopupOrResultPopup("input");
    setAge("");
    setHeight("");
    setWeight("");
  }

  useEffect(() => {
    const fetchRecommendedSize = async () => {
      setAPIResponseMsg(await getRecommendedSize(age, height, weight));
    };

    fetchRecommendedSize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [APIRequestBody]);

  return (
    <>
      {inputPopupOrResultPopup === "input" ? (
        <form className="input-popup">
          <h3>Enter Your Data</h3>
          <div className="input-field">
            <label htmlFor="age">Age:</label>
            <input
              type="text"
              ref={ageInput}
              value={age}
              onChange={handleAgeChange}
            />
          </div>
          <div className="input-field">
            <label htmlFor="height">Height:</label>
            <input
              type="text"
              ref={heightInput}
              value={height}
              onChange={handleHeightChange}
            />
          </div>
          <div className="input-field">
            <label htmlFor="weight">Weight:</label>
            <input
              type="text"
              ref={weightInput}
              value={weight}
              onChange={handleWeightChange}
            />
          </div>

          <span ref={errorLabel}></span>

          <div className="buttons">
            <button className="close" onClick={handleFormCancelation}>
              Close
            </button>
            <button className="submit" onClick={handleFormSubmission}>
              Submit
            </button>
          </div>
        </form>
      ) : (
        <div className="result-popup">
          <h1>Result</h1>

          <p>
            Based on the data provided earlier, we've did some calculations on
            this data and we got this
          </p>

          <h3>{APIResponseMsg}</h3>

          <div className="buttons">
            <button className="close" onClick={handleFormCancelation}>
              Close
            </button>
            <button className="enter-again" onClick={handleEnterAgainBtn}>
              Again
            </button>
          </div>
        </div>
      )}
    </>
  );
}
