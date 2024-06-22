import React, { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  selectMale,
  selectFemale,
} from "../../utils/loginAndSignUpFunctions/selectGender";
import {
  selectCustomer,
  selectSeller,
} from "../../utils/loginAndSignUpFunctions/selectUserType";
import { isFormValid } from "../../utils/loginAndSignUpFunctions/signUpValidations";

import "../../components/LandingComponents/styles/Sections.css";

const handleSubmit = (e) => {
  e.preventDefault();

  // Add your sign up logic
};

export default function SignUp({ setIsUserSignedIn }) {
  const navigate = useNavigate();

  const maleImg = useRef(null);
  const femaleImg = useRef(null);
  const customerImg = useRef(null);
  const sellerImg = useRef(null);
  const policyCheckBoxElement = useRef(null);

  const [userGender, setUserGender] = useState("");
  const [nameValue, setNameValue] = useState("");
  const [emailValue, setEmailValue] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [passwordValue, setPasswordValue] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [userType, setUserType] = useState("");
  const [policyCheckBox, setPolicyCheckBox] = useState(false);

  return (
    <>
      <section className="section_1">
        <div className="left-side">
          <h1>
            Registration Form
            <span role="img" aria-label="emoji">
              😁
            </span>
          </h1>

          <p>Please, fill out your information.</p>

          <br />

          <form
            action=""
            method=""
            className="signUpForm"
            onSubmit={handleSubmit}
          >
            <div>
              <img
                src={"/resources/images/png/male.png"}
                alt="male"
                className="genderImg"
                ref={maleImg}
                onClick={() => selectMale(maleImg, femaleImg, setUserGender)}
              />
              <img
                src={"/resources/images/png/female.png"}
                alt="female"
                className="genderImg"
                ref={femaleImg}
                onClick={() => selectFemale(maleImg, femaleImg, setUserGender)}
              />
            </div>

            <div>
              <img
                src={"/resources/images/gif/name.gif"}
                alt="name"
                className="inputFieldImg"
              />
              <input
                type="text"
                name="name"
                placeholder="Full name"
                className="nameInput"
                value={nameValue}
                onChange={(e) => setNameValue(e.target.value)}
                required
              />
            </div>

            <div>
              <img
                src={"/resources/images/gif/phone.gif"}
                alt="phone"
                className="inputFieldImg"
              />
              <input
                type="text"
                name="phone"
                placeholder="Phone Number"
                className="phoneInput"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                required
              />
            </div>

            <div>
              <img
                src={"/resources/images/gif/email-file.gif"}
                alt="email"
                className="inputFieldImg"
              />
              <input
                type="email"
                name="email"
                placeholder="Email"
                className="emailInput"
                value={emailValue}
                onChange={(e) => setEmailValue(e.target.value)}
                required
              />
            </div>

            <div>
              <img
                src={"/resources/images/gif/password.gif"}
                alt="password"
                className="inputFieldImg"
              />
              <input
                type="password"
                name="password"
                placeholder="Password"
                className="passInput"
                value={passwordValue}
                onChange={(e) => setPasswordValue(e.target.value)}
                required
              />
            </div>

            <div>
              <img
                src={"/resources/images/gif/confirm.gif"}
                alt="confirmPass"
                className="inputFieldImg"
              />
              <input
                type="password"
                name="confirmPass"
                placeholder="Confirm password"
                className="confirmPassInput"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>

            <div>
              <img
                src={"/resources/images/png/customer.png"}
                alt="customer"
                className="userType"
                ref={customerImg}
                onClick={() =>
                  selectCustomer(customerImg, sellerImg, setUserType)
                }
              />
              <img
                src={"/resources/images/png/seller.png"}
                alt="seller"
                className="userType"
                ref={sellerImg}
                onClick={() =>
                  selectSeller(customerImg, sellerImg, setUserType)
                }
              />
            </div>

            <div>
              <input
                ref={policyCheckBoxElement}
                type="checkbox"
                name="policies"
                id="policies"
                className="policiesInput"
                onChange={() => {
                  setPolicyCheckBox(policyCheckBoxElement.current.checked);
                }}
              />
              <label htmlFor="policies" className="checkBoxLabel">
                {" "}
                By creating an account you agree to <br /> our Terms of Service
                and Privacy Policy.
              </label>
            </div>

            {isFormValid(
              userGender,
              nameValue,
              emailValue,
              phoneNumber,
              passwordValue,
              confirmPassword,
              userType,
              policyCheckBox
            ) ? (
              <button
                type="submit"
                onClick={() => {
                  setIsUserSignedIn(true);
                  navigate("/");
                }}
              >
                Sign up
              </button>
            ) : (
              <button className="disabledBtn" disabled>
                Sign up
              </button>
            )}

            <Link className="alreadyHaveAcc" to="/login">
              Already have an account? Login!
            </Link>

            <div>
              <img
                src={"/resources/images/png/google.png"}
                alt="login"
                className="loginAPIs"
              />
              <img
                src={"/resources/images/png/facebook.png"}
                alt="login"
                className="loginAPIs"
              />
              <img
                src={"/resources/images/png/apple.png"}
                alt="login"
                className="loginAPIs"
              />
            </div>
          </form>
        </div>

        <div className="right-side">
          <img
            src={"/resources/images/svg/signUp.svg"}
            alt="signUp"
            loading="lazy"
          />
        </div>
      </section>
    </>
  );
}
