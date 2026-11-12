import Foundation
import CoreLocation

struct LocationResult: Codable {
    let status: String
    let latitude: Double?
    let longitude: Double?
    let accuracy_m: Double?
    let city: String?
    let error: String?
    let source: String?
}

final class LocationFetcher: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private var isDone = false
    private let timeoutSeconds: TimeInterval

    init(timeoutSeconds: TimeInterval = 12.0) {
        self.timeoutSeconds = timeoutSeconds
        super.init()
        self.manager.delegate = self
        self.manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
    }

    func start() {
        guard CLLocationManager.locationServicesEnabled() else {
            finish(status: "services_disabled", error: "macOS Location Services are disabled globally")
            return
        }

        let status = manager.authorizationStatus
        handleAuthorizationStatus(status)
    }

    func handleAuthorizationStatus(_ status: CLAuthorizationStatus) {
        switch status {
        case .notDetermined:
            // Trigger native macOS TCC prompt
            manager.requestWhenInUseAuthorization()
        case .authorizedAlways, .authorizedWhenInUse:
            manager.startUpdatingLocation()
        case .restricted:
            finish(status: "restricted", error: "Location Services restricted by system policy")
        case .denied:
            finish(status: "denied", error: "Location permission denied by user")
        @unknown default:
            finish(status: "unknown", error: "Unknown authorization status")
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        if status != .notDetermined {
            handleAuthorizationStatus(status)
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }

        // Filter out invalid/negative accuracy readings
        if location.horizontalAccuracy < 0 {
            return
        }

        manager.stopUpdatingLocation()

        let lat = location.coordinate.latitude
        let lon = location.coordinate.longitude
        let acc = location.horizontalAccuracy

        // Reverse geocode to city name using native CLGeocoder
        geocoder.reverseGeocodeLocation(location) { [weak self] placemarks, _ in
            var cityName: String? = nil
            if let p = placemarks?.first {
                if let locality = p.locality, let country = p.country {
                    cityName = "\(locality), \(country)"
                } else if let subAdmin = p.subAdministrativeArea, let country = p.country {
                    cityName = "\(subAdmin), \(country)"
                } else if let country = p.country {
                    cityName = country
                }
            }

            self?.finish(
                status: "success",
                latitude: lat,
                longitude: lon,
                accuracy_m: acc,
                city: cityName,
                source: "macos_native"
            )
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        if let clErr = error as? CLError, clErr.code == .denied {
            finish(status: "denied", error: "Location access denied")
        } else {
            finish(status: "error", error: error.localizedDescription)
        }
    }

    private func finish(
        status: String,
        latitude: Double? = nil,
        longitude: Double? = nil,
        accuracy_m: Double? = nil,
        city: String? = nil,
        error: String? = nil,
        source: String? = nil
    ) {
        guard !isDone else { return }
        isDone = true

        let result = LocationResult(
            status: status,
            latitude: latitude,
            longitude: longitude,
            accuracy_m: accuracy_m,
            city: city,
            error: error,
            source: source
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        if let data = try? encoder.encode(result), let jsonStr = String(data: data, encoding: .utf8) {
            print(jsonStr)
            fflush(stdout)
        } else {
            print("{\"error\":\"Serialization failure\",\"status\":\"error\"}")
            fflush(stdout)
        }
    }

    func run() {
        start()
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while !isDone && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.1))
        }

        if !isDone {
            finish(status: "timeout", error: "Location request timed out after \(Int(timeoutSeconds)) seconds")
        }
    }
}

@main
struct LocationHelperApp {
    static func main() {
        var timeout: TimeInterval = 12.0
        if CommandLine.arguments.count > 1, let parsed = Double(CommandLine.arguments[1]), parsed > 0 {
            timeout = parsed
        }

        let fetcher = LocationFetcher(timeoutSeconds: timeout)
        fetcher.run()
    }
}

