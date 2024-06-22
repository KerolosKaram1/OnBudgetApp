export const getAllOrders = async () => {
  return [
    {
      orderNum: "#asdf@13573",
      customerID: "23ad",
      customerName: "John Doe",
      date: "2022-01-01",
      total: 100.0,
      status: "shipped",
    },
    {
      orderNum: "#vafb@23",
      customerID: "23ad",
      customerName: "John Doe",
      date: "2022-01-01",
      total: 100.0,
      status: "arrived",
    },
    {
      orderNum: "#axcavv@4234",
      customerID: "23ad",
      customerName: "John Doe",
      date: "2022-01-01",
      total: 100.0,
      status: "delivered",
    },
    {
      orderNum: "#fseef@364",
      customerID: "23ad",
      customerName: "John Doe",
      date: "2022-01-01",
      total: 100.0,
      status: "canceled",
    },
    {
      orderNum: "#da@12357",
      customerID: "23ad",
      customerName: "John Doe",
      date: "2022-01-01",
      total: 100.0,
      status: "canceled and returned",
    },
  ];

  // return [];
};
