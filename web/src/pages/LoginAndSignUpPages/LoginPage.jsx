import React, { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { showPass } from "../../utils/loginAndSignUpFunctions/showPass";
import { isFormValid } from "../../utils/loginAndSignUpFunctions/loginValidations";

import "../../components/LandingComponents/styles/Sections.css";

export default function Login({ setIsUserSignedIn }) {
  const navigate = useNavigate();
  const passInput = useRef(null);
  const rememberMeCheckBox = useRef(null);

  const [emailOrPhone, setEmailOrPhone] = useState("");
  const [passwordValue, setPasswordValue] = useState("");

  // eslint-disable-next-line no-unused-vars
  const [isRememberMeChecked, setIsRememberMeChecked] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();

    // Add your login logic
  };

  return (
    <>
      <section className="section_1">
        <div className="left-side">
          <h1>
            Welcome Back!
            <span role="img" aria-label="emoji">
              🥰
            </span>
          </h1>

          <p>Sign in using your email or phone number.</p>

          <br />

          <form
            action=""
            method=""
            className="loginForm"
            onSubmit={handleSubmit}
          >
            <div>
              <img
                src={"/resources/images/gif/email-file.gif"}
                alt="email"
                className="inputFieldImg"
              />
              <input
                type="text"
                name="emailOrPhone"
                placeholder="Email or Phone"
                className="emailInput"
                value={emailOrPhone}
                onChange={(e) => setEmailOrPhone(e.target.value)}
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
                ref={passInput}
                value={passwordValue}
                onChange={(e) => setPasswordValue(e.target.value)}
                required
              />

              <br />
              <br />

              <label htmlFor="showPass" className="checkBoxLabel">
                <input
                  type="checkbox"
                  name="showPass"
                  id="showPass"
                  className="showPassInput"
                  onChange={() => showPass(passInput)}
                />{" "}
                Show password
              </label>
            </div>

            {isFormValid(emailOrPhone, passwordValue) ? (
              <button
                type="submit"
                onClick={() => {
                  setIsUserSignedIn(true);
                  navigate("/");
                }}
              >
                Login
              </button>
            ) : (
              <button className="disabledBtn" disabled>
                Login
              </button>
            )}

            <div>
              <label htmlFor="rememberMe" className="checkBoxLabel">
                <input
                  ref={rememberMeCheckBox}
                  type="checkbox"
                  name="rememberMe"
                  id="rememberMe"
                  className="rememberMeInput"
                  onChange={() => {
                    setIsRememberMeChecked(rememberMeCheckBox.current.checked);
                  }}
                />{" "}
                Remember Me
              </label>
              <br />
              <br />
              <span>Forget password?</span>
              <br /> <br />
              <Link className="register" to="/signup">
                Don't have an account? Sign up!
              </Link>
            </div>

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
            src={"/resources/images/svg/login.svg"}
            alt="login"
            loading="lazy"
          />
        </div>
      </section>
    </>
  );
}
