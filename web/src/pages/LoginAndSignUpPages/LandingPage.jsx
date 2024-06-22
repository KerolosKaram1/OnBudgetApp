/* eslint-disable react/jsx-pascal-case */
import React, { useState } from "react";
import Section_1 from "./../../components/LandingComponents/Section_1.jsx";
import Section_2 from "./../../components/LandingComponents/Section_2.jsx";
import Section_3 from "./../../components/LandingComponents/Section_3.jsx";
import Section_4 from "./../../components/LandingComponents/Section_4.jsx";

export default function Landing() {
  const [section, setSection] = useState("section_1");

  return (
    <>
      {section === "section_1" && (
        <Section_1 section={section} setSection={setSection} />
      )}
      {section === "section_2" && (
        <Section_2 section={section} setSection={setSection} />
      )}
      {section === "section_3" && (
        <Section_3 section={section} setSection={setSection} />
      )}
      {section === "section_4" && <Section_4 />}
    </>
  );
}
