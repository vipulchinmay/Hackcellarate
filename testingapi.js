const apiKey = 'YOUR_API_KEY'; // Replace with your actual API key
const latitude = 28.6139;  // Example: New Delhi center
const longitude = 77.2090;
const radiusInMeters = 8000; // 8 km
const type = 'doctor';

const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${latitude},${longitude}&radius=${radiusInMeters}&type=${type}&key=${apiKey}`;

fetch(url)
  .then(response => response.json())
  .then(data => {
    console.log("Nearby Doctors:", data.results);
    data.results.forEach((place, index) => {
      console.log(`${index + 1}. ${place.name} - ${place.vicinity}`);
    });
  })
  .catch(error => {
    console.error("Error fetching doctor locations:", error);
  });
