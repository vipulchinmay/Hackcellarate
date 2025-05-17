const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const twilio = require("twilio");
require("dotenv").config(); // Load environment variables

const app = express();

// Enable CORS with specific origins
app.use(cors({ origin: process.env.FRONTEND_URL || "http://localhost:3000" }));
app.use(express.json());

// Set up storage for uploaded images
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    if (file.fieldname === "photo") {
      if (file.mimetype === "image/png" || file.mimetype === "image/jpg" || file.mimetype === "image/jpeg") {
        cb(null, true);
      } else {
        cb(null, false);
        return cb(new Error("Only .png, .jpg and .jpeg format allowed!"));
      }
    } else {
      cb(null, true);
    }
  },
});

// Serve static files from the uploads directory
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// Connect to MongoDB
mongoose
  .connect(process.env.MONGODB_URI || "mongodb://localhost:27017/vnr", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.log("❌ MongoDB Connection Error:", err));

// Define User Schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  age: { type: String, required: true },
  gender: { type: String, required: true },
  blood_group: { type: String, required: true },
  medical_conditions: String,
  date_of_birth: { type: Date, required: true },
  photo: { type: String }, // Store the path to the photo
  created_at: { type: Date, default: Date.now },
  updated_at: { type: Date, default: Date.now },
});

const User = mongoose.model("User", userSchema);

// Define Reminder Schema for medicine reminders
const reminderSchema = new mongoose.Schema({
  medicineName: {
    type: String,
    required: true,
  },
  scheduledTime: {
    type: Date,
    required: true,
  },
  day: {
    type: Number,
    required: true,
  },
  taken: {
    type: Boolean,
    default: false,
  },
  notificationId: String,
  createdAt: {
    type: Date,
    default: Date.now,
  },
  userId: String, // Link to user ID when auth is implemented
});

const Reminder = mongoose.model("Reminder", reminderSchema);

// Twilio credentials
const twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);

// ===== USER PROFILE ROUTES =====

// Save or Update Profile
app.post("/profile", upload.single("photo"), async (req, res) => {
  try {
    const { _id, name, age, gender, blood_group, medical_conditions, date_of_birth } = req.body;

    // Basic Validation
    if (!name || !age || !gender || !blood_group || !date_of_birth) {
      return res.status(400).json({ success: false, message: "Missing required fields!" });
    }

    // Prepare user data
    const userData = {
      name,
      age,
      gender,
      blood_group,
      medical_conditions,
      date_of_birth,
      updated_at: new Date(),
    };

    // Add photo path if a photo was uploaded
    if (req.file) {
      userData.photo = `/uploads/${req.file.filename}`;
    }

    let user;
    if (_id) {
      // Update Existing User
      user = await User.findByIdAndUpdate(_id, userData, { new: true });
    } else {
      // Create New User
      userData.created_at = new Date();
      user = new User(userData);
      await user.save();
    }

    res.json({ success: true, message: "Profile saved successfully!", user });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ success: false, message: "Internal Server Error" });
  }
});

// Fetch Profile by ID
app.get("/profile/:id", async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }
    res.json({ success: true, user });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ success: false, message: "Internal Server Error" });
  }
});

// Fetch Profile by Name
app.get("/profile/name/:name", async (req, res) => {
  try {
    const user = await User.findOne({ name: req.params.name });
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }
    res.json({ success: true, user });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ success: false, message: "Internal Server Error" });
  }
});

// Fetch All Profiles
app.get("/profiles", async (req, res) => {
  try {
    const users = await User.find();
    res.json({ success: true, users });
  } catch (err) {
    res.status(500).json({ success: false, message: "Internal Server Error" });
  }
});

// Delete Profile
app.delete("/profile/:id", async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }

    // Delete associated photo if it exists
    if (user.photo) {
      const photoPath = path.join(__dirname, user.photo);
      if (fs.existsSync(photoPath)) {
        fs.unlinkSync(photoPath);
      }
    }

    await User.findByIdAndDelete(req.params.id);
    res.json({ success: true, message: "Profile deleted successfully!" });
  } catch (err) {
    res.status(500).json({ success: false, message: "Internal Server Error" });
  }
});

// Send SMS with User Info
app.post("/getno", async (req, res) => {
  try {
    const { userData, phoneNumber } = req.body;

    if (!userData || !phoneNumber) {
      return res.status(400).json({ success: false, message: "Missing user data or phone number." });
    }

    // Construct the SMS message
    const message = `User Details:\nName: ${userData.name}\nAge: ${userData.age}\nGender: ${userData.gender}\nBlood Group: ${userData.blood_group}\nDate of Birth: ${userData.date_of_birth}\nMedical Conditions: ${userData.medical_conditions || "None"}`;

    // Send SMS using Twilio
    await twilioClient.messages.create({
      body: message,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: phoneNumber,
    });

    res.json({ success: true, message: "SMS sent successfully!" });
  } catch (error) {
    console.error("Error sending SMS:", error);
    res.status(500).json({ success: false, message: "Failed to send SMS." });
  }
});

// ===== MEDICINE REMINDER ROUTES =====

// Create and list medicine reminders
app.get("/api/medicine/schedule", async (req, res) => {
  try {
    // Get all reminders (in a production app, you'd filter by user ID)
    const reminders = await Reminder.find({})
      .sort({ scheduledTime: 1 })
      .lean();
    
    res.status(200).json({ success: true, data: reminders });
  } catch (error) {
    console.error('Error fetching reminders:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch reminders' });
  }
});

app.post("/api/medicine/schedule", async (req, res) => {
  try {
    const { medicineName, scheduledTime, day } = req.body;

    console.log("Hello");
    // Basic validation
    if (!medicineName || !scheduledTime) {
      return res.status(400).json({ 
        success: false, 
        error: "Medicine name and scheduled time are required" 
      });
    }

    // Validate scheduledTime (ensure it is a valid date/time)
    if (isNaN(new Date(scheduledTime).getTime())) {
      return res.status(400).json({
        success: false,
        error: "Invalid scheduled time format",
      });
    }

    // Create reminder in the database
    const reminder = await Reminder.create({
      medicineName,
      scheduledTime: new Date(scheduledTime), // Store as Date object
      day: day || null, // Default to null if not provided
      taken: false,
      //userId: req.user._id, // Uncomment when authentication is implemented
    });

    res.status(201).json({ success: true, id: reminder._id });
  } catch (error) {
    console.error("Error creating reminder:", error);
    res.status(500).json({ success: false, error: "Failed to create reminder" });
  }
});

app.put("/api/medicine/schedule/:id", async (req, res) => {
  try {
    // Update reminder by ID
    const updateData = req.body;
    
    const reminder = await Reminder.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true, runValidators: true }
    );
    
    if (!reminder) {
      return res.status(404).json({ success: false, error: 'Reminder not found' });
    }
    
    res.status(200).json({ success: true, data: reminder });
  } catch (error) {
    console.error('Error updating reminder:', error);
    res.status(500).json({ success: false, error: 'Failed to update reminder' });
  }
});

app.delete("/api/medicine/schedule/:id", async (req, res) => {
  try {
    // Delete reminder by ID
    const deletedReminder = await Reminder.findByIdAndDelete(req.params.id);
    
    if (!deletedReminder) {
      return res.status(404).json({ success: false, error: 'Reminder not found' });
    }
    
    res.status(200).json({ success: true, data: {} });
  } catch (error) {
    console.error('Error deleting reminder:', error);
    res.status(500).json({ success: false, error: 'Failed to delete reminder' });
  }
});

// Mark medicine as taken
app.post("/api/medicine/taken", async (req, res) => {
  try {
    const { notificationId } = req.body;
    
    if (!notificationId) {
      return res.status(400).json({ success: false, error: 'Notification ID is required' });
    }
    
    // Find reminder by notification ID and mark as taken
    const reminder = await Reminder.findOneAndUpdate(
      { notificationId },
      { taken: true },
      { new: true }
    );
    
    if (!reminder) {
      return res.status(404).json({ 
        success: false, 
        error: 'Reminder not found for the given notification ID' 
      });
    }
    
    // Return success response
    res.status(200).json({ 
      success: true, 
      message: `${reminder.medicineName} marked as taken`,
      data: reminder
    });
  } catch (error) {
    console.error('Error marking medicine as taken:', error);
    res.status(500).json({ success: false, error: 'Failed to mark medicine as taken' });
  }
});

// Get active (not taken) reminders for a user
app.get("/api/medicine/active", async (req, res) => {
  try {
    // In a real app, you'd get userId from authentication
    // const userId = req.user._id;
    
    const activeReminders = await Reminder.find({
      // userId: userId,
      taken: false,
      scheduledTime: { $gte: new Date() }
    }).sort({ scheduledTime: 1 });
    
    res.status(200).json({ success: true, data: activeReminders });
  } catch (error) {
    console.error('Error fetching active reminders:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch active reminders' });
  }
});

// Get medication history (taken reminders)
app.get("/api/medicine/history", async (req, res) => {
  try {
    // In a real app, you'd get userId from authentication
    // const userId = req.user._id;
    
    const history = await Reminder.find({
      // userId: userId,
      taken: true
    }).sort({ scheduledTime: -1 });
    
    res.status(200).json({ success: true, data: history });
  } catch (error) {
    console.error('Error fetching medication history:', error);
    res.status(500).json({ success: false, error: 'Failed to fetch medication history' });
  }
});

// Send medicine reminder SMS
app.post("/api/medicine/remind-sms", async (req, res) => {
  try {
    const { phoneNumber, medicineName, scheduledTime } = req.body;

    if (!phoneNumber || !medicineName) {
      return res.status(400).json({ 
        success: false, 
        error: 'Phone number and medicine name are required' 
      });
    }

    // Format time for readability
    const formattedTime = new Date(scheduledTime).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });

    // Construct the SMS message
    const message = `REMINDER: It's time to take your ${medicineName} at ${formattedTime}. Stay healthy!`;

    // Send SMS using Twilio
    await twilioClient.messages.create({
      body: message,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: phoneNumber,
    });

    res.json({ success: true, message: "Reminder SMS sent successfully!" });
  } catch (error) {
    console.error("Error sending reminder SMS:", error);
    res.status(500).json({ success: false, error: "Failed to send reminder SMS" });
  }
});

// Start the server
const PORT = process.env.PORT || 6000;
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));