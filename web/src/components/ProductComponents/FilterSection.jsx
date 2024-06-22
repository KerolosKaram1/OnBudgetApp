import {
  Box,
  Collapse,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
} from "@mui/material";

export default function FilterSection({
  filterOpen,
  filterCriteria,
  handleFilterChange,
  minPrice,
  maxPrice,
  handleMaxPriceChange,
  handleMinPriceChange,
}) {
  return (
    <Collapse in={filterOpen} timeout="auto" unmountOnExit>
      <Box sx={{ marginTop: "1rem", marginBottom: "2rem" }}>
        {/* Sort By */}
        <FormControl fullWidth>
          <InputLabel id="filter-select-label">Filter By</InputLabel>
          <Select
            labelId="filter-select-label"
            id="filter-select"
            value={filterCriteria}
            label="Filter By"
            onChange={handleFilterChange}
          >
            <MenuItem value="priceLowToHigh">Price: Low to High</MenuItem>
            <MenuItem value="priceHighToLow">Price: High to Low</MenuItem>
            <MenuItem value="rating">Rating</MenuItem>
            <MenuItem value="newest">Newest</MenuItem>
          </Select>
        </FormControl>

        {/* Price Limit */}
        <Box sx={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
          <TextField
            label="Min Price"
            type="number"
            value={minPrice}
            onChange={handleMinPriceChange}
            fullWidth
          />
          <TextField
            label="Max Price"
            type="number"
            value={maxPrice}
            onChange={handleMaxPriceChange}
            fullWidth
          />
        </Box>
      </Box>
    </Collapse>
  );
}
