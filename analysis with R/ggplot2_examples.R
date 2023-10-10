library(ggplot2)
library(palmerpenguins)
data("penguins")
ggplot(data=penguins) + 
  geom_jitter(mapping = aes(x = flipper_length_mm,
                           y = body_mass_g,
                           color = species))
ggplot(data=penguins) + 
  geom_smooth(mapping = aes(x = flipper_length_mm,
                           y = body_mass_g, linetype = species)) +
  geom_point(mapping = aes(x = flipper_length_mm,
                           y = body_mass_g))

data("diamonds")
ggplot(date= diamonds) +
  geom_bar(mapping = aes(x=cut, color = cut))


ggplot(data=penguins) + 
  geom_smooth(mapping = aes(x = flipper_length_mm,
                            y = body_mass_g, linetype = species)) +

# facets
ggplot(data=penguins, aes(x=flipper_length_mm, y=body_mass_g)) +
  geom_point(aes(color=species)) +
  facet_grid(sex~species) + 
  labs(title = "table name", subtitle =  "subtitle name", caption =  "caption")
