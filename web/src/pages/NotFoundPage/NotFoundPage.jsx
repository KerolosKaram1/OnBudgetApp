export default function NotFoundPage() {
  return (
    <main
      style={{
        maxWidth: "100%",
        maxHeight: "100vh",
        display: "flex",
        flexFlow: "column nowrap",
      }}
    >
      <div style={{ padding: "5rem 1rem 0 5rem" }}>
        <img
          src={"resources/images/notFound.png"}
          alt="not found"
          style={{ maxWidth: "100%" }}
        />
        <h1 style={{ textAlign: "center", paddingTop: "5rem" }}>
          Sorry, We can't find this page.😥
        </h1>
      </div>
    </main>
  );
}
