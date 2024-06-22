import React from "react";
import "./styles/Sections.css";

export default function Section_3({ section, setSection }) {
  return (
    <section className="section_3">
      <div className="left-side">
        <h1>
          Door To Door Delivery
          <span role="img" aria-label="emoji">
            😁
          </span>
        </h1>

        <p>
          Get your shopping done quickly just a few taps even if you had a small{" "}
          <span>Budget</span>.
          <br />
          We've variety of delivery options.
        </p>

        <button onClick={() => setSection("section_2")}>Back</button>
        <button onClick={() => setSection("section_4")}>Join Us</button>

        <div className="dots">
          <div className="dot1"></div>
          <div className="dot2"></div>
          <div
            className="dot3"
            style={{ backgroundColor: "rgb(255, 205, 78)" }}
          ></div>
        </div>
      </div>

      <div className="right-side">
        <img
          src={"/resources/images/svg/delivery.svg"}
          alt="delivery"
          loading="lazy"
        />
      </div>
    </section>
  );
}
