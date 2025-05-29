import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Alert,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Linking,
  Dimensions,
  LogBox,
  Modal,
  ScrollView,
} from "react-native";
import * as Location from "expo-location";
import MapView, { Marker, PROVIDER_GOOGLE } from "react-native-maps";

// Ignore specific warnings if needed
// LogBox.ignoreLogs(['Warning: ...']);

const { width, height } = Dimensions.get("window");
const ASPECT_RATIO = width / height;
const LATITUDE_DELTA = 0.02;
const LONGITUDE_DELTA = LATITUDE_DELTA * ASPECT_RATIO;

// Replace with your actual API key safely (use environment variables in production)
const GOOGLE_MAPS_API_KEY = "AIzaSyCEib-Wk5PZyn7pqH0HlCsv3ek8DWGinzg";

// Mock available appointment slots for the next 3 days
const generateTimeSlots = () => {
  const slots = [];
  const today = new Date();
  
  // Generate slots for the next 3 days
  for (let day = 0; day < 3; day++) {
    const date = new Date(today);
    date.setDate(today.getDate() + day);
    const dateStr = date.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
    
    // Morning slots
    for (let hour = 9; hour < 12; hour++) {
      slots.push({
        id: `${dateStr}-${hour}:00`,
        date: dateStr,
        time: `${hour}:00 AM`,
        available: Math.random() > 0.3, // 70% chance of availability
      });
      slots.push({
        id: `${dateStr}-${hour}:30`,
        date: dateStr,
        time: `${hour}:30 AM`,
        available: Math.random() > 0.3,
      });
    }
    
    // Afternoon slots
    for (let hour = 1; hour < 5; hour++) {
      slots.push({
        id: `${dateStr}-${hour}:00PM`,
        date: dateStr,
        time: `${hour}:00 PM`,
        available: Math.random() > 0.3,
      });
      slots.push({
        id: `${dateStr}-${hour}:30PM`,
        date: dateStr,
        time: `${hour}:30 PM`,
        available: Math.random() > 0.3,
      });
    }
  }
  
  return slots;
};

export default function DoctorConsultScreen() {
  const [location, setLocation] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mapRegion, setMapRegion] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  
  // Appointment booking states
  const [bookingModalVisible, setBookingModalVisible] = useState(false);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [timeSlots, setTimeSlots] = useState([]);
  const [selectedDate, setSelectedDate] = useState(null);
  const [appointmentConfirmed, setAppointmentConfirmed] = useState(false);
  const [selectedTimeSlot, setSelectedTimeSlot] = useState(null);

  useEffect(() => {
    fetchHospitals();
  }, []);

  const fetchHospitals = async () => {
    setLoading(true);
    setErrorMsg(null);
    
    try {
      // Request location permission
      const { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== "granted") {
        setErrorMsg("Permission to access location was denied");
        Alert.alert("Permission Denied", "Allow location access to find hospitals.");
        setLoading(false);
        return;
      }
      
      // Get current location
      const userLocation = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });
      
      console.log("User location obtained:", userLocation.coords);
      setLocation(userLocation);

      const { latitude, longitude } = userLocation.coords;
      
      // Set map region
      const region = {
        latitude,
        longitude,
        latitudeDelta: LATITUDE_DELTA,
        longitudeDelta: LONGITUDE_DELTA,
      };
      
      setMapRegion(region);
      console.log("Map region set:", region);

      // Fetch nearby hospitals
      const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${latitude},${longitude}&radius=8000&type=hospital&key=${GOOGLE_MAPS_API_KEY}`;
      console.log("Fetching hospitals from:", url);
      
      const response = await fetch(url);
      const data = await response.json();
      
      console.log("API Response status:", data.status);
      
      if (data.status !== "OK") {
        throw new Error(`API Error: ${data.status} - ${data.error_message || "Unknown error"}`);
      }

      if (data.results && data.results.length > 0) {
        console.log(`Found ${data.results.length} hospitals`);
        
        const enrichedHospitals = await Promise.all(
          data.results.map(async (hospital) => {
            try {
              const detailsUrl = `https://maps.googleapis.com/maps/api/place/details/json?place_id=${hospital.place_id}&fields=name,vicinity,formatted_phone_number,rating,geometry&key=${GOOGLE_MAPS_API_KEY}`;
              const detailsResponse = await fetch(detailsUrl);
              const detailsData = await detailsResponse.json();
              
              if (detailsData.result) {
                return {
                  ...detailsData.result,
                  distance: calculateDistance(
                    latitude,
                    longitude,
                    detailsData.result.geometry.location.lat,
                    detailsData.result.geometry.location.lng
                  ),
                };
              }
              return null;
            } catch (error) {
              console.error("Error fetching hospital details:", error);
              return null;
            }
          })
        );

        const validHospitals = enrichedHospitals.filter((h) => h !== null);
        console.log(`Processed ${validHospitals.length} valid hospitals`);
        setHospitals(validHospitals);
      } else {
        console.log("No hospitals found in the area");
        setHospitals([]);
      }
    } catch (error) {
      console.error("Error in fetchHospitals:", error);
      setErrorMsg(error.message);
      Alert.alert("Error", `Failed to fetch hospital details: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371; // Radius of the Earth in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return (R * c).toFixed(2);
  };

  const makeCall = (phoneNumber) => {
    if (!phoneNumber) {
      Alert.alert("No Phone Number", "This hospital doesn't have a phone number listed.");
      return;
    }
    Linking.openURL(`tel:${phoneNumber}`);
  };

  const openDirections = (destLat, destLng) => {
    if (!location) {
      Alert.alert("Location Error", "Your current location is not available.");
      return;
    }
    
    const { latitude, longitude } = location.coords;
    const url = `https://www.google.com/maps/dir/?api=1&origin=${latitude},${longitude}&destination=${destLat},${destLng}&travelmode=driving`;
    Linking.openURL(url);
  };

  // Function to retry location access
  const retryLocationAccess = () => {
    fetchHospitals();
  };

  // Appointment booking functions
  const openBookingModal = (hospital) => {
    setSelectedHospital(hospital);
    setTimeSlots(generateTimeSlots());
    setSelectedDate(null);
    setSelectedTimeSlot(null);
    setAppointmentConfirmed(false);
    setBookingModalVisible(true);
  };

  const closeBookingModal = () => {
    setBookingModalVisible(false);
    setSelectedHospital(null);
    setSelectedDate(null);
    setSelectedTimeSlot(null);
    setAppointmentConfirmed(false);
  };

  const selectDate = (date) => {
    setSelectedDate(date);
    setSelectedTimeSlot(null);
  };

  const bookAppointment = () => {
    if (!selectedTimeSlot) {
      Alert.alert("Selection Required", "Please select a time slot for your appointment.");
      return;
    }

    // In a real app, this would make an API call to book the appointment
    console.log("Booking appointment:", {
      hospital: selectedHospital.name,
      date: selectedTimeSlot.date,
      time: selectedTimeSlot.time,
    });

    // Show confirmation
    setAppointmentConfirmed(true);
    
    // In a real app, you might navigate to a confirmation screen after a delay
    setTimeout(() => {
      closeBookingModal();
      Alert.alert(
        "Appointment Confirmed",
        `Your appointment at ${selectedHospital.name} on ${selectedTimeSlot.date} at ${selectedTimeSlot.time} has been booked.`
      );
    }, 2000);
  };

  // Get unique dates from time slots
  const getUniqueDates = () => {
    const dates = [...new Set(timeSlots.map(slot => slot.date))];
    return dates;
  };

  // Filter time slots by selected date
  const getTimeSlotsForDate = () => {
    if (!selectedDate) return [];
    return timeSlots.filter(slot => slot.date === selectedDate);
  };

  // Appointment Booking Modal
  const renderBookingModal = () => {
    const uniqueDates = getUniqueDates();
    const filteredTimeSlots = getTimeSlotsForDate();
    
    return (
      <Modal
        animationType="slide"
        transparent={true}
        visible={bookingModalVisible}
        onRequestClose={closeBookingModal}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            {appointmentConfirmed ? (
              <View style={styles.confirmationContainer}>
                <Text style={styles.confirmationTitle}>Appointment Confirmed!</Text>
                <Text style={styles.confirmationText}>
                  Your appointment at {selectedHospital?.name} has been scheduled for{" "}
                  {selectedTimeSlot?.date} at {selectedTimeSlot?.time}.
                </Text>
                <ActivityIndicator size="large" color="#4CAF50" style={styles.confirmationSpinner} />
              </View>
            ) : (
              <>
                <View style={styles.modalHeader}>
                  <Text style={styles.modalTitle}>Book Appointment</Text>
                  <TouchableOpacity onPress={closeBookingModal} style={styles.closeButton}>
                    <Text style={styles.closeButtonText}>✕</Text>
                  </TouchableOpacity>
                </View>
                
                <Text style={styles.hospitalTitle}>{selectedHospital?.name}</Text>
                <Text style={styles.hospitalAddress}>{selectedHospital?.vicinity}</Text>
                
                <ScrollView style={styles.modalScrollView}>
                  {/* Date Selection */}
                  <Text style={styles.sectionTitle}>Select Date:</Text>
                  <View style={styles.dateContainer}>
                    {uniqueDates.map((date) => (
                      <TouchableOpacity
                        key={date}
                        style={[
                          styles.dateButton,
                          selectedDate === date && styles.selectedDateButton,
                        ]}
                        onPress={() => selectDate(date)}
                      >
                        <Text
                          style={[
                            styles.dateButtonText,
                            selectedDate === date && styles.selectedDateButtonText,
                          ]}
                        >
                          {date}
                        </Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                  
                  {/* Time Slots Selection */}
                  {selectedDate ? (
                    <>
                      <Text style={styles.sectionTitle}>Select Time:</Text>
                      <View style={styles.timeSlotContainer}>
                        {filteredTimeSlots.map((slot) => (
                          <TouchableOpacity
                            key={slot.id}
                            style={[
                              styles.timeSlotButton,
                              !slot.available && styles.unavailableTimeSlot,
                              selectedTimeSlot?.id === slot.id && styles.selectedTimeSlot,
                            ]}
                            onPress={() => slot.available && setSelectedTimeSlot(slot)}
                            disabled={!slot.available}
                          >
                            <Text
                              style={[
                                styles.timeSlotText,
                                !slot.available && styles.unavailableTimeSlotText,
                                selectedTimeSlot?.id === slot.id && styles.selectedTimeSlotText,
                              ]}
                            >
                              {slot.time}
                            </Text>
                          </TouchableOpacity>
                        ))}
                      </View>
                    </>
                  ) : (
                    <Text style={styles.instructionText}>Please select a date first</Text>
                  )}
                </ScrollView>
                
                <TouchableOpacity
                  style={[
                    styles.bookButton,
                    !selectedTimeSlot && styles.disabledButton,
                  ]}
                  onPress={bookAppointment}
                  disabled={!selectedTimeSlot}
                >
                  <Text style={styles.bookButtonText}>Confirm Appointment</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>
      </Modal>
    );
  };

  return (
    <View style={styles.container}>
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2196F3" />
          <Text style={styles.loadingText}>Finding nearby hospitals...</Text>
        </View>
      ) : errorMsg ? (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{errorMsg}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={retryLocationAccess}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          {mapRegion && (
            <MapView 
              provider={PROVIDER_GOOGLE}
              style={styles.map} 
              region={mapRegion}
              showsUserLocation={true}
              showsMyLocationButton={true}
            >
              {/* User location marker */}
              <Marker 
                coordinate={{
                  latitude: mapRegion.latitude,
                  longitude: mapRegion.longitude,
                }}
                title="Your Location"
                pinColor="#2196F3"
              />
              
              {/* Hospital markers */}
              {hospitals.map((hospital, index) => (
                hospital.geometry && hospital.geometry.location ? (
                  <Marker
                    key={`hospital-${index}`}
                    coordinate={{
                      latitude: hospital.geometry.location.lat,
                      longitude: hospital.geometry.location.lng,
                    }}
                    title={hospital.name}
                    description={hospital.vicinity}
                    pinColor="#ff3b30"
                  />
                ) : null
              ))}
            </MapView>
          )}

          <View style={styles.listContainer}>
            <Text style={styles.listTitle}>
              {hospitals.length > 0 
                ? `Found ${hospitals.length} Nearby Hospitals` 
                : "No hospitals found nearby"}
            </Text>
            
            <FlatList
              data={hospitals}
              keyExtractor={(item, index) => `hospital-list-${index}`}
              renderItem={({ item }) =>
                item && item.name && item.vicinity ? (
                  <View style={styles.card}>
                    <Text style={styles.hospitalName}>{item.name}</Text>
                    <Text style={styles.text}>📍 {item.vicinity}</Text>
                    {item.formatted_phone_number && (
                      <Text style={styles.text}>📞 {item.formatted_phone_number}</Text>
                    )}
                    {item.rating && <Text style={styles.text}>⭐ Rating: {item.rating}/5</Text>}
                    {item.distance && <Text style={styles.text}>📏 Distance: {item.distance} km</Text>}
                    
                    {/* Button Container */}
                    <View style={styles.buttonContainer}>
                      <TouchableOpacity
                        style={styles.callButton}
                        onPress={() => makeCall(item.formatted_phone_number)}
                        disabled={!item.formatted_phone_number}
                      >
                        <Text style={styles.buttonText}>Call</Text>
                      </TouchableOpacity>
                      
                      <TouchableOpacity
                        style={styles.directionButton}
                        onPress={() => openDirections(
                          item.geometry.location.lat, 
                          item.geometry.location.lng
                        )}
                      >
                        <Text style={styles.buttonText}>Directions</Text>
                      </TouchableOpacity>
                      
                      <TouchableOpacity
                        style={styles.appointmentButton}
                        onPress={() => openBookingModal(item)}
                      >
                        <Text style={styles.buttonText}>Book</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                ) : null
              }
              ListEmptyComponent={
                <View style={styles.emptyListContainer}>
                  <Text style={styles.emptyListText}>No hospitals found in your area</Text>
                  <TouchableOpacity style={styles.retryButton} onPress={retryLocationAccess}>
                    <Text style={styles.retryButtonText}>Retry Search</Text>
                  </TouchableOpacity>
                </View>
              }
            />
          </View>
          
          {renderBookingModal()}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#25292e",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    color: "#fff",
    marginTop: 10,
    fontSize: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  errorText: {
    color: "#ff3b30",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: "#2196F3",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
  },
  retryButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  map: {
    width: Dimensions.get("window").width,
    height: height * 0.4,
  },
  listContainer: {
    flex: 1,
    padding: 10,
  },
  listTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
    textAlign: "center",
  },
  card: {
    backgroundColor: "#333",
    padding: 15,
    marginVertical: 8,
    borderRadius: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  hospitalName: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 5,
  },
  text: {
    color: "#fff",
    fontSize: 16,
    marginBottom: 5,
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 10,
  },
  callButton: {
    flex: 1,
    marginRight: 5,
    backgroundColor: "#4CAF50",
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  directionButton: {
    flex: 1,
    marginRight: 5,
    backgroundColor: "#2196F3",
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  appointmentButton: {
    flex: 1,
    backgroundColor: "#9C27B0",
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  emptyListContainer: {
    padding: 20,
    alignItems: "center",
  },
  emptyListText: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
  },
  
  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  modalContent: {
    backgroundColor: "#333",
    borderRadius: 15,
    width: "100%",
    maxHeight: height * 0.8,
    padding: 20,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 15,
  },
  modalTitle: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "bold",
  },
  closeButton: {
    padding: 5,
  },
  closeButtonText: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "bold",
  },
  hospitalTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 5,
  },
  hospitalAddress: {
    color: "#ccc",
    fontSize: 16,
    marginBottom: 20,
  },
  modalScrollView: {
    marginBottom: 20,
  },
  sectionTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
    marginTop: 10,
  },
  dateContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 20,
  },
  dateButton: {
    backgroundColor: "#444",
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 10,
    marginRight: 10,
    marginBottom: 10,
  },
  selectedDateButton: {
    backgroundColor: "#2196F3",
  },
  dateButtonText: {
    color: "#fff",
    fontSize: 14,
  },
  selectedDateButtonText: {
    fontWeight: "bold",
  },
  timeSlotContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 10,
  },
  timeSlotButton: {
    backgroundColor: "#444",
    paddingVertical: 12,
    paddingHorizontal: 15,
    borderRadius: 10,
    marginRight: 10,
    marginBottom: 10,
    width: "30%",
    alignItems: "center",
  },
  unavailableTimeSlot: {
    backgroundColor: "#555",
    opacity: 0.5,
  },
  selectedTimeSlot: {
    backgroundColor: "#4CAF50",
  },
  timeSlotText: {
    color: "#fff",
    fontSize: 14,
  },
  unavailableTimeSlotText: {
    color: "#999",
  },
  selectedTimeSlotText: {
    fontWeight: "bold",
  },
  instructionText: {
    color: "#ccc",
    fontSize: 16,
    textAlign: "center",
    marginTop: 20,
    marginBottom: 20,
  },
  bookButton: {
    backgroundColor: "#4CAF50",
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  disabledButton: {
    backgroundColor: "#555",
    opacity: 0.7,
  },
  bookButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  
  // Confirmation styles
  confirmationContainer: {
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  confirmationTitle: {
    color: "#4CAF50",
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 15,
    textAlign: "center",
  },
  confirmationText: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
  },
  confirmationSpinner: {
    marginTop: 20,
    marginBottom: 10,
  },
});