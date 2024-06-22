import React from "react";
import "./styles/Sections.css";

export default function Section_2({ section, setSection }) {
  return (
    <section className="section_2">
      <div className="left-side">
        <h1>
          Easy Track Order
          <span role="img" aria-label="emoji">
            😉
          </span>
        </h1>

        <p>
          Choose your favorite shipping option, Which suits your{" "}
          <span>Budget</span>.
          <br />
          We've the fastest delivery service.
        </p>

        <button onClick={() => setSection("section_1")}>Back</button>
        <button onClick={() => setSection("section_3")}>Next</button>

        <div className="dots">
          <div className="dot1"></div>
          <div
            className="dot2"
            style={{ backgroundColor: "rgb(255, 205, 78)" }}
          ></div>
          <div className="dot3"></div>
        </div>
      </div>

      <div className="right-side">
        <img
          src={"/resources/images/svg/trackOrder.svg"}
          alt="trackOrder"
          loading="lazy"
        />
      </div>
    </section>
  );
}
