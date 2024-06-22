import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getAllCategories } from "./../../utils/getAllCategories";
import { Container, Typography, Divider, Grid, Box } from "@mui/material";

export default function CategoriesPage() {
  const [categories, setCategories] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchCategories = async () => {
      setCategories(await getAllCategories());
    };

    fetchCategories();
  }, []);

  return (
    <main
      style={{
        marginTop: "5rem",
        marginLeft: "5rem",
      }}
    >
      <Container style={{ paddingBottom: "3rem" }}>
        <Typography variant="h4" gutterBottom>
          All Categories 🤩
        </Typography>

        <Divider sx={{ marginBottom: "2rem" }} />

        <Grid container spacing={4}>
          {categories.map((category) => (
            <Grid
              item
              xs={12}
              sm={6}
              md={4}
              lg={3}
              key={category.id}
              style={{ marginBottom: "1rem" }}
            >
              <Box
                sx={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: 2,
                  border: "1px solid #ccc",
                  borderRadius: "8px",
                  cursor: "pointer",
                  boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
                  "&:hover": {
                    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                  },
                }}
                onClick={() => {
                  navigate(`/categories/${category.title}`);
                }}
              >
                <img
                  src={category.image}
                  alt={category.title}
                  style={{ width: "8rem", height: "8rem", borderRadius: "50%" }}
                />
                <Typography variant="subtitle1" sx={{ marginTop: "1rem" }}>
                  {category.title}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Container>
    </main>
  );
}
